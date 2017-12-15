#########################################################
# This code is a port of TISK 1.x Distribution by Heejo You.
# This is not core file of TISK 1.x Distribution,
# but it is the example codes for using that.
# For the detail meaning of each command,
# see https://github.com/CODEJIN/Tisk or You & Magnuson (2018).
#########################################################

# load the TISK functions
import Basic_TISK_Class as tisk

# load the phoneme and pronunciation [word] lists and
# prepare appropriate connections
phoneme_List, pronunciation_List = tisk.List_Generate()

#you can change the lexicon file by assigning the parameter 'pronunciation_File'.
# phoneme_List, pronunciation_List = tisk.List_Generate(
#                             pronunciation_File = 'other_lexicon.txt')

# initialize the model with the the phoneme_List, pronunciation_List, number of time slots, and threshold
tisk_Model = tisk.TISK_Model(phoneme_List, pronunciation_List,
                             time_Slots = 10,
                             nPhone_Threshold = 0.91)



# initialize the model with default or current parameters
tisk_Model.Weight_Initialize()


# change selected TISK parameters
tisk_Model.Decay_Parameter_Assign(
                    decay_Phoneme = 0.001,
                    decay_Diphone = 0.001,
                    decay_SPhone = 0.001,
                    decay_Word = 0.01)
tisk_Model.Weight_Parameter_Assign(
                    input_to_Phoneme_Weight = 1.0,
                    phoneme_to_Phone_Weight = 0.1,
                    diphone_to_Word_Weight = 0.05,
                    sPhone_to_Word_Weight = 0.01,
                    word_to_Word_Weight = -0.005)
tisk_Model.Feedback_Parameter_Assign(
                    word_to_Diphone_Activation = 0,
                    word_to_SPhone_Activation = 0,
                    word_to_Diphone_Inhibition = 0,
                    word_to_SPhone_Inhibition = 0)
tisk_Model.Weight_Initialize()


# select specific paramters
tisk_Model.Decay_Parameter_Assign(
                    decay_Phoneme = 0.002,
                    decay_Diphone = 0.0005)

#To see the list the current parameters
tisk_Model.Parameter_Display()


# trigger a simulation without producing output;
# this prepares a model for inspection
tisk_Model.Display_Graph(pronunciation='pat')


# trigger a simulation and create a phoneme input graph
tisk_Model.Display_Graph(
           pronunciation='pat',
           display_Phoneme_List = [('p', 0), ('a', 1), ('t', 2)])

# trigger a simulation and make 3 graphs
tisk_Model.Display_Graph(pronunciation='pat',
                         display_Diphone_List = ['pa', 'pt', 'ap'],
                         display_Single_Phone_List = ['p', 'a', 't'],
                         display_Word_List = ['pat', 'tap'])

# trigger a simulation, make 3 graphs, save them as PNG files
tisk_Model.Display_Graph(pronunciation='pat',
                         display_Diphone_List = ['pa', 'pt', 'ap'],
                         display_Single_Phone_List = ['p', 'a', 't'],
                         display_Word_List = ['pat', 'tap'],
                         file_Save = True)


# trigger a simulation and ready data structures
# without creating any output
tisk_Model.Extract_Data(pronunciation='pat')

# trigger a simulation and assign data structures to 'result'
result = tisk_Model.Extract_Data(pronunciation='pat',
                                 extract_Word_List = ['pat', 'tap'])

# trigger a simulation and assign data structures to 'result'
result = tisk_Model.Extract_Data(pronunciation='pat',
            extract_Phoneme_List = [('p', 0), ('a', 1), ('t', 2)],
            extract_Single_Phone_List = ['p', 'a', 't'])

# trigger a simulation and save data structures to text file
result = tisk_Model.Extract_Data(pronunciation='pat',
            extract_Word_List = ['pat', 'tap'], file_Save=True)

# get mean RT and accuracy for the specified set of words
rt_and_ACC = tisk_Model.Run_List(
            pronunciation_List = ['baks','bar','bark','bat^l','bi'])

# get mean RT and accuracy for all words in pronunciation_List
rt_and_ACC = tisk_Model.Run_List(
            pronunciation_List = pronunciation_List)


# get mean RT and accuracy for specified word list with
# specified accuracy criteria for abs, rel, and tim, respectively
rt_and_ACC = tisk_Model.Run_List(
            pronunciation_List = ['baks','bar','bark','bat^l','bi'],
            absolute_Acc_Criteria=0.6,
            relative_Acc_Criteria=0.01,
            time_Acc_Criteria=5)


#Export reaction time
tisk_Model.Run_List(
            pronunciation_List = ['baks','bar','bark','bat^l','bi'],
            output_File_Name = "Test",
            reaction_Time=True)


tisk_Model.Run_List(
            pronunciation_List = pronunciation_List,
            output_File_Name = "all",
            reaction_Time=True)



# get mean RT and accuracy for specified word list
# with specified accuracy criteria but ALSO
# save activation histories in 'raw' and 'category' formats
rt_and_ACC = tisk_Model.Run_List(
            pronunciation_List = ['baks','bar','bark','bat^l','bi'],
            output_File_Name = 'Result',
            raw_Data = True,
            categorize=True)

# trigger a simulation and make a graph
tisk_Model.Average_Activation_by_Category_Graph(
            pronunciation_List=['baks'])


tisk_Model.Average_Activation_by_Category_Graph(
            pronunciation_List=['pat', 'tap', 'art'])


# trigger a simulation, make a graph, save them as PNG files
tisk_Model.Average_Activation_by_Category_Graph(
            pronunciation_List=['pat','tap', 'art'],
            file_Save=True)


# trigger a simulation, make a graph, save them as PNG files
tisk_Model.Average_Activation_by_Category_Graph(
            pronunciation_List=['pat','tap', 'art'],
            output_File_Name='Result',
            file_Save=True)


# make a graph for all words in pronunciation_List
tisk_Model.Average_Activation_by_Category_Graph(
            pronunciation_List = pronunciation_List)


# Getting the competitor information
competitor_List = tisk_Model.Category_List(
            pronunciation = 'b^s')

# Display cohort list
print(competitor_List[0])

# Display rhyme count
print(len(competitor_List[1]))

# Display the mean competitor count
tisk_Model.Display_Mean_Category_Count(
            pronunciation_List = pronunciation_List)


## Batch size control
rt_and_ACC = tisk_Model.Run_List(
            pronunciation_List = pronunciation_List,
            batch_Size = 10)


## Reaction time and accuracy for specific words
result = tisk_Model.Run('pat')
abs_RT = tisk_Model.RT_Absolute_Threshold(
            pronunciation = 'pat',
            word_Activation_Array = result[3],
            criterion = 0.75)
rel_RT = tisk_Model.RT_Relative_Threshold(
            pronunciation = 'pat',
            word_Activation_Array = result[3],
            criterion = 0.05)
tim_RT = tisk_Model.RT_Time_Dependent(
            pronunciation = 'pat',
            word_Activation_Array = result[3],
            criterion = 10)


rt_and_ACC = tisk_Model.Run_List(pronunciation_List = ['pat'])


# first, select all words that have length 3 in the lexicon
length3_Pronunciation_List = [x for x in pronunciation_List if len(x) == 3]

# now do the same for words with length 5
length5_Pronunciation_List = [x for x in pronunciation_List if len(x) == 5]

# make a graph of average competitor effects for 3-phoneme words
tisk_Model.Average_Activation_by_Category_Graph(
            pronunciation_List = length3_Pronunciation_List)

# make a graph and also save to a PNG file
tisk_Model.Average_Activation_by_Category_Graph(
            pronunciation_List= length3_Pronunciation_List,
            file_Save=True,
            output_File_Name='length_3_category_results.png')

# make a graph and also save to a PNG file
tisk_Model.Average_Activation_by_Category_Graph(
            pronunciation_List= length5_Pronunciation_List,
            file_Save=True,
            output_File_Name= 'length_5_category_results.png')

# save the length 3 data
tisk_Model.Run_List( pronunciation_List = length3_Pronunciation_List,
            output_File_Name='length3data',
            raw_Data = True, categorize = True)

# check competitor counts of length 3 and length 5 lists

tisk_Model.Display_Mean_Category_Count(
            pronunciation_List = length3_Pronunciation_List)

tisk_Model.Display_Mean_Category_Count(
            pronunciation_List = length3_Pronunciation_List)


# Select words with length greater than or equal to 3 with
# phoneme /a/ in position 2
filtered_Pronunciation_List = [x for x in pronunciation_List if len(x) >= 3 and x[2] == 'a']

# make a graph
tisk_Model.Average_Activation_by_Category_Graph(
            pronunciation_List = filtered_Pronunciation_List)

# select words with length greater than or equal to 3 with
# phoneme /a/ in position 2 and phoneme /k/ in position 3
filtered_Pronunciation_List = [x for x in pronunciation_List if len(x) >= 3 and x[2] == 'a' and x[3]=='k']
