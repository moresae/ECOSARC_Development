# main script to run CNN pipeline according to settings specified in 0_pipeline_settings.py
# this script handles data labelling, import, and any transformations



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GET METADATA

# get train and test dfs
case_df_train , video_df_train , frame_df_train = import_df( bucket_train )
case_df_test , video_df_test , frame_df_test = import_df( bucket_test )

#print(len(case_df_train),len(video_df_train),len(frame_df_train))
#print(len(case_df_test),len(video_df_test),len(frame_df_test))

# get KPA values as floats
case_df_train = KPA_floats(case_df_train)
video_df_train = KPA_floats(video_df_train)
frame_df_train = KPA_floats(frame_df_train)
case_df_test = KPA_floats(case_df_test)
video_df_test = KPA_floats(video_df_test)
frame_df_test = KPA_floats(frame_df_test)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LABEL CASES
if labelling == 1:

    # apply labels to train
    case_df_train , video_df_train , frame_df_train = label_disease_1( case_df_train , video_df_train , frame_df_train , min_kpa )
    # apply labels to test
    case_df_test , video_df_test , frame_df_test = label_disease_1( case_df_test , video_df_test , frame_df_test , min_kpa )

if labelling == 2:
    
    # apply labels to train
    case_df_train , video_df_train , frame_df_train = label_disease_2( case_df_train , video_df_train , frame_df_train , h_c , f0_c , f2_c , f3_c , f4_c , c_c )
    # apply labels to test
    case_df_test , video_df_test , frame_df_test = label_disease_2( case_df_test , video_df_test , frame_df_test , h_c , f0_c , f2_c , f3_c , f4_c , c_c )

if labelling == 3:

    # apply labels to train
    case_df_train , video_df_train , frame_df_train = label_disease_3( case_df_train , video_df_train , frame_df_train 
                                                                      , h_c , f0_c , f2_c, f3_c, f4_c , f_noKPA , c_c , other_c , other_cases )

    # apply labels to test
    case_df_test , video_df_test , frame_df_test = label_disease_3( case_df_test , video_df_test , frame_df_test 
                                                                     , h_c , f0_c , f2_c, f3_c, f4_c , f_noKPA , c_c , other_c , other_cases )

if labelling == 4:

    # apply labels to train
    case_df_train, video_df_train, frame_df_train = label_disease_4(case_df_train, video_df_train, frame_df_train
                                                                    , s1_cut, s2_cut, s3_cut, s0_c, s1_c, s2_c, s3_c, f_c, h_c)

    # apply labels to test
    case_df_test, video_df_test, frame_df_test = label_disease_4(case_df_test, video_df_test, frame_df_test
                                                                    , s1_cut, s2_cut, s3_cut, s0_c, s1_c, s2_c, s3_c, f_c, h_c)




# remove unwanted cases (anything with label '-1')
print('-train and val:')
case_df_train , video_df_train , frame_df_train = remove_bad_cases( case_df_train , video_df_train , frame_df_train )
print('-test:')
case_df_test , video_df_test , frame_df_test = remove_bad_cases( case_df_test , video_df_test , frame_df_test )

numClasses = len( case_df_train['Label'].value_counts() )
#print(numClasses)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# REMOVE OR KEEP SHADOWY FRAMES

if remove_shadowing:
    # remove shadowing
    print('-train and val:')
    frame_df_train = remove_shadow_frame( frame_df_train )
    print('-test:')
    frame_df_test = remove_shadow_frame( frame_df_test )

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GET FRAME FILENAMES

# only necessary for sequential loading, but inconsequential otherwise
frame_df_train = get_frame_names( frame_df_train , data_type , bucket_train_name , path_to_main )
frame_df_test = get_frame_names( frame_df_test , data_type , bucket_test_name , path_to_main )

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# train/val/test split

# shuffle order of cases for a unique split of cases between train and val
if trainval_shuffle:
    frame_df_train = organize_df( frame_df_train , shuffle_cases = True )

if use_existing_split:
    print('\tusing cases from existing split, from dataset: ' , dataset_name )
    train_IDs_t = np.unique( np.asarray( pd.read_pickle( path_to_main+'saved_datasets/'+dataset_name + '/train.pkl' )['caseID'].tolist() ) )
    val_IDs_t = np.unique( np.asarray( pd.read_pickle( path_to_main+'saved_datasets/'+dataset_name + '/val.pkl' )['caseID'].tolist() ) )
    
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    
    for ii in range(0,len(train_IDs_t)):
        train_df = pd.concat( [ train_df , frame_df_train[ frame_df_train['caseID'] == train_IDs_t[ii] ] ] )
    
    for ii in range(0,len(val_IDs_t)):
        val_df = pd.concat( [ val_df , frame_df_train[ frame_df_train['caseID'] == val_IDs_t[ii] ] ] )
elif use_exact_existing_split:
    print('\tusing exact split from dataset: ' , dataset_name )
    
    train_df = pd.read_pickle( (path_to_main+'saved_datasets/'+dataset_name + '/train.pkl') )
    val_df = pd.read_pickle( (path_to_main+'saved_datasets/'+dataset_name + '/val.pkl') )
    
    train_df = get_frame_names( train_df , data_type , bucket_train_name , path_to_main )
    val_df = get_frame_names( val_df , data_type , bucket_train_name , path_to_main )
    
else:
    if numTotal_train_val == -1:
        numTotal_train_val = len(frame_df_train)
    # split it
    if train_even and train_duplicate:
        train_df , val_df = train_val_split( frame_df_train , numTotal_train_val , split_frac , numClasses , False , val_even )
        train_df = balance_duplicate( train_df , numClasses )
    else:
        train_df , val_df = train_val_split( frame_df_train , numTotal_train_val , split_frac , numClasses , train_even , val_even )

# shuffle here unless there are multiple instances per frame (in that case, shuffling is done automatically later)
if ( instance_rate == 1 and ( not use_exact_existing_split ) ):
    train_df , new_train_order , seed_used = shuffle_df_seeded( train_df , use_random_seed , chosen_seed )
    print( 'shuffled train_df using seed =' , seed_used )
    #train_df = train_df.reindex(drop=True)

# test split
test_df = frame_df_test
if test_even:
    test_df = balance_classes( test_df , numClasses , False )
    
# print summary
print('train set:')
df_label_summary(train_df)
print('val set:')
df_label_summary(val_df)
print('test set:')
df_label_summary(test_df)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DATA IMPORT and modification, if applicable. Labels and IDs are also extracted for training and testing.
# if instance_rate = 1, data should be transformed immediately and put straight into arrays, rather than first being stored as lists

if train_val_RAM:
    
    print('Starting import.')
    
    if instance_rate == 1:
        # import and transform if necessary
        train0_test1 = 0
        #train_data = import_data_array( train_df , data_type , data_shape , modify_function , train0_test1 , local , num_cores )
        train_data = import_data_array_FAST( train_df , data_type , data_shape , modify_function , train0_test1 , local , num_cores )
        gc.collect()
        train0_test1 = 0
        #val_data = import_data_array( val_df , data_type , data_shape , modify_function , train0_test1 , local , num_cores )
        val_data = import_data_array_FAST( val_df , data_type , data_shape , modify_function , train0_test1 , local , num_cores )
        gc.collect()
    else:
        # import and transform if necessary
        # shuffling is done in the import function automatically
        train0_test1 = 0
        train_data , train_df = import_data_array_FAST_2( instance_rate , train_df , data_type , data_shape , modify_function , train0_test1 , local , num_cores, ran=26 )
        gc.collect()
        train0_test1 = 0
        val_data , val_df = import_data_array_FAST_2( instance_rate , val_df , data_type , data_shape , modify_function , train0_test1 , local , num_cores , ran=26)
        gc.collect()
        
        # extend metadata dataframe for test
        # train and val were already extended and shuffled in the import function above
        test_df = extend_df( test_df , instance_rate )
        gc.collect()
        
        ## import
        #train0_test1 = 0
        #train_data = import_data_list( train_df , data_type , train0_test1 , local , num_cores )
        #gc.collect()
        #train0_test1 = 1
        #val_data = import_data_list( val_df , data_type , train0_test1 , local , num_cores )
        #gc.collect()
        ## extend metadata dataframes to reflect true number of training instances
        #train_df = extend_df( train_df , instance_rate )
        #val_df = extend_df( val_df , instance_rate )
        #test_df = extend_df( test_df , instance_rate )
        #gc.collect()
        ## transform/modify the data
        #train_data = modify_data_list( train_data , instance_rate , modify_function )
        #gc.collect()
        #val_data = modify_data_list( val_data , instance_rate , modify_function )
        #gc.collect()
        ## reshuffle train data and dataframe together
        #train_df , train_data = shuffle_df_data( train_df , train_data )
        #gc.collect()
        ## switch from lists to arrays (this can be memory intensive)
        #train_data = np.asarray(train_data)
        #gc.collect()
        #gc.collect()
        #val_data = np.asarray(val_data)
        #gc.collect()
        #gc.collect()
    
    # get labels, dictionaries
    IDs_train , labels_train , labelDict_train = labels_dicts( train_df )
    IDs_val , labels_val , labelDict_val = labels_dicts( val_df )
    IDs_test , labels_test , labelDict_test = labels_dicts( test_df )
    # get 1hot labels
    labels_train_1hot = one_hot_encode( numClasses , labels_train )
    labels_val_1hot = one_hot_encode( numClasses , labels_val )
    # set validation tuple
    val_tuple = ( val_data , labels_val_1hot )
    
    print('Import Done.')
    
else:
    if instance_rate != 1:
        # extend metadata dataframes to reflect true number of training instances
        train_df = extend_df( train_df , instance_rate )
        val_df = extend_df( val_df , instance_rate )
        test_df = extend_df( test_df , instance_rate )
    # get labels, dictionaries
    IDs_train , labels_train , labelDict_train = labels_dicts( train_df )
    IDs_val , labels_val , labelDict_val = labels_dicts( val_df )
    IDs_test , labels_test , labelDict_test = labels_dicts( test_df )
    # get 1hot labels
    labels_train_1hot = one_hot_encode( numClasses , labels_train )
    labels_val_1hot = one_hot_encode( numClasses , labels_val )
    
    print('Sequential Dataprep complete.')
    
gc.collect()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SAVE DATASET (if requested)
# saves the data which has already been loaded into RAM (and possibly transformed) for more convenient use later, while keeping the same train/val/test split

if save_datasets and train_val_RAM:
    
    # datetime(year, month, day, hour, minute, second, microsecond)
    right_now = datetime.datetime.now()
    date_s = ( '{}_{}_{}_{}_{}'.format(right_now.year,right_now.month,right_now.day,right_now.hour,right_now.minute) )
    date_s
    # name dataset
    dataset_name = data_type + '_dataset_' + date_s
    
    # define dataframe to save labelling scheme and other metadata
    r_df = pd.DataFrame()
    
    # basic data
    r_df['data type'] = [data_type]
    r_df['data shape'] = [data_shape]
    
    # modify function
    r_df['modify_function'] = [modify_function]
    
    # labelling data
    r_df['numClasses'] = numClasses
    r_df['labelling'] = labelling
    r_df['instance_rate'] = instance_rate
    if labelling == 1:
        r_df['min_kpa'] = min_kpa
    elif labelling == 2:
        r_df['h_c'] = h_c
        r_df['f0_c'] = f0_c
        r_df['f2_c'] = f2_c
        r_df['f3_c'] = f3_c
        r_df['f4_c'] = f4_c
        r_df['c_c'] = c_c
    elif labelling == 3:
        r_df['h_c'] = h_c
        r_df['f0_c'] = f0_c
        r_df['f2_c'] = f2_c
        r_df['f3_c'] = f3_c
        r_df['f4_c'] = f4_c
        r_df['f_noKPA'] = f_noKPA
        r_df['c_c'] = c_c
        r_df['other_c'] = other_c
    
    # summary of train/val/test split
    r_df['train size'] = [len(train_df)]
    r_df['train split'] = [np.asarray( train_df['Label'].value_counts() )]
    r_df['val size'] = [len(val_df)]
    r_df['val split'] = [np.asarray( val_df['Label'].value_counts() )]
    try:
        r_df['test size'] = [len(test_df)]
        r_df['test split'] = [np.asarray( test_df['Label'].value_counts() )]
    except:
        r_df['test size'] = [np.nan]
        r_df['test split'] = [np.nan]
    
    # save dataframes and data into folder
    dataset_name = dataset_name + custom_save_label
    os.mkdir( path_to_main+'saved_datasets/'+dataset_name )
    r_df.to_pickle( (path_to_main+'saved_datasets/'+dataset_name + '/info.pkl') )
    train_df.to_pickle( (path_to_main+'saved_datasets/'+dataset_name + '/train.pkl') )
    val_df.to_pickle( (path_to_main+'saved_datasets/'+dataset_name + '/val.pkl') )
    test_df.to_pickle( (path_to_main+'saved_datasets/'+dataset_name + '/test.pkl') )
    
    np.save( (path_to_main+'saved_datasets/'+dataset_name  + '/train.npy') , train_data )
    np.save( (path_to_main+'saved_datasets/'+dataset_name + '/val.npy') , val_data )
    
else:
    # no dataset has been loaded
    print('no dataset will be saved')

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print('----------DONE----------')