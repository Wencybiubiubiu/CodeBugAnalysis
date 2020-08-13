import _pickle as pickle
import os
import time

import numpy as np
import shutil
import tensorflow as tf

import reader as reader
from common import Common
from extractor import Extractor

from rouge import FilesRouge
import math
import random


class Model:
    topk = 10
    num_batches_to_log = 100

    def __init__(self, config):
        self.K = 12 # 12 types of exception
        self.config = config
        self.sess = tf.Session()

        self.eval_queue = None
        self.predict_queue = None

        self.eval_placeholder = None
        self.predict_placeholder = None
        self.eval_predicted_indices_op, self.eval_top_values_op, self.eval_true_target_strings_op, self.eval_topk_values = None, None, None, None
        self.predict_top_indices_op, self.predict_top_scores_op, self.predict_target_strings_op = None, None, None
        self.subtoken_to_index = None

        self.path_source_indices_matrix, self.node_indices_matrix, self.path_target_indices_matrix = None, None, None
        self.source_word_embed_matrix, self.path_embed_matrix, self.target_word_embed_matrix  = None, None, None
        self.target_string = None
        self.source_string, self.path_string, self.target_string = None, None, None

        self.predict_path_source_indices_matrix, self.predict_node_indices_matrix, self.predict_path_target_indices_matrix = None, None, None
        self.predict_source_word_embed_matrix, self.predict_path_embed_matrix, self.predict_target_word_embed_matrix  = None, None, None

        self.index_of_initial_center = None
        self.saved_center_source, self.saved_center_path, self.saved_center_target = [], [], []
        self.group_for_each_data = None
        self.predict_group_for_each_data = None

        self.num_of_iter = 200

        self.subtoken_vocab, self.target_words_vocab, self.nodes_vocab = None, None, None

        self.subtoken_table, self.target_table, self.get_node_table = None, None, None

        if config.LOAD_PATH:
            self.load_model(sess=None)
        #elif config.PREDICT:
        #    pass
        else:
            with open('{}.dict.c2s'.format(config.TRAIN_PATH), 'rb') as file:
                subtoken_to_count = pickle.load(file)
                node_to_count = pickle.load(file)
                target_to_count = pickle.load(file)
                max_contexts = pickle.load(file)
                self.num_training_examples = pickle.load(file)
                print('Dictionaries loaded.')

            if self.config.DATA_NUM_CONTEXTS <= 0:
                self.config.DATA_NUM_CONTEXTS = max_contexts
            self.subtoken_to_index, self.index_to_subtoken, self.subtoken_vocab_size = \
                Common.load_vocab_from_dict(subtoken_to_count, add_values=[Common.PAD, Common.UNK],
                                            max_size=config.SUBTOKENS_VOCAB_MAX_SIZE)
            print('Loaded subtoken vocab. size: %d' % self.subtoken_vocab_size)

            self.target_to_index, self.index_to_target, self.target_vocab_size = \
                Common.load_vocab_from_dict(target_to_count, add_values=[Common.PAD, Common.UNK, Common.SOS],
                                            max_size=config.TARGET_VOCAB_MAX_SIZE)
            print('Loaded target word vocab. size: %d' % self.target_vocab_size)

            self.node_to_index, self.index_to_node, self.nodes_vocab_size = \
                Common.load_vocab_from_dict(node_to_count, add_values=[Common.PAD, Common.UNK], max_size=None)
            print('Loaded nodes vocab. size: %d' % self.nodes_vocab_size)
            self.epochs_trained = 0

    def close_session(self):
        self.sess.close()

    def train(self):
        print('Starting training')
        start_time = time.time()

        batch_num = 0
        sum_loss = 0
        best_f1 = 0
        best_epoch = 0
        best_f1_precision = 0
        best_f1_recall = 0
        epochs_no_improve = 0

        self.queue_thread = reader.Reader(subtoken_to_index=self.subtoken_to_index,
                                          node_to_index=self.node_to_index,
                                          target_to_index=self.target_to_index,
                                          config=self.config)

        print("\nProcessing data: \n")
        self.get_the_processed_data(self.queue_thread.get_output())
        print("Processed data loaded.\n")

        print("Begin training:\n")
        for i in range(self.num_of_iter):
            self.sort_all_data()
            self.update_center()

        print(self.group_for_each_data)

        self.check_null_group()

        #self.save_sorted_groups()
        #self.load_sorted_groups()

        #test = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
        #self.save_centers(test, 'test.txt')
        #self.load_centers('test.txt')

        #self.save_all_centers()
        #self.load_all_centers()

        self.predict()

    def get_the_processed_data(self, input_tensors):
        source_string = input_tensors[reader.PATH_SOURCE_STRINGS_KEY]
        path_string = input_tensors[reader.PATH_STRINGS_KEY]
        target_string = input_tensors[reader.PATH_TARGET_STRINGS_KEY]
        target_string = input_tensors[reader.TARGET_STRING_KEY]
        target_index = input_tensors[reader.TARGET_INDEX_KEY]
        target_lengths = input_tensors[reader.TARGET_LENGTH_KEY]
        path_source_indices = input_tensors[reader.PATH_SOURCE_INDICES_KEY]
        node_indices = input_tensors[reader.NODE_INDICES_KEY]
        path_target_indices = input_tensors[reader.PATH_TARGET_INDICES_KEY]
        valid_context_mask = input_tensors[reader.VALID_CONTEXT_MASK_KEY]
        path_source_lengths = input_tensors[reader.PATH_SOURCE_LENGTHS_KEY]
        path_lengths = input_tensors[reader.PATH_LENGTHS_KEY]
        path_target_lengths = input_tensors[reader.PATH_TARGET_LENGTHS_KEY]

        with tf.variable_scope('model'):
            subtoken_vocab = tf.get_variable('SUBTOKENS_VOCAB',
                                             shape=(self.subtoken_vocab_size, self.config.EMBEDDINGS_SIZE),
                                             dtype=tf.float32,
                                             initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                        mode='FAN_OUT',
                                                                                                        uniform=True))
            target_words_vocab = tf.get_variable('TARGET_WORDS_VOCAB',
                                                 shape=(self.target_vocab_size, self.config.EMBEDDINGS_SIZE),
                                                 dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                            mode='FAN_OUT',
                                                                                                            uniform=True))
            nodes_vocab = tf.get_variable('NODES_VOCAB', shape=(self.nodes_vocab_size, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32,
                                          initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                     mode='FAN_OUT',
                                                                                                     uniform=True))
            # (batch, max_contexts, decoder_size)
            source_word_embed,  path_embed, target_word_embed = self.compute_contexts(subtoken_vocab=subtoken_vocab, nodes_vocab=nodes_vocab,
                                                     source_input=path_source_indices, nodes_input=node_indices,
                                                     target_input=path_target_indices,
                                                     valid_mask=valid_context_mask,
                                                     path_source_lengths=path_source_lengths,
                                                     path_lengths=path_lengths, path_target_lengths=path_target_lengths)

            self.subtoken_vocab, self.target_words_vocab, self.nodes_vocab = subtoken_vocab,target_words_vocab,nodes_vocab

            batch_size = tf.shape(target_index)[0]

        #Here, the program gives us the processed data-sets
        #every context, in the definition, has the form of [source, path, target]
        # path_target_indices is the index matrix for all targets
        # node_indices is the index matrix for all paths
        # path_source_indices is the index matrix for all sources
        # source_word_embed is the embedding for path_source_indices
        # path_embed is the embedding for node_indices
        # target_word_embed is the embedding for path_target_indices

        #now we just transfer all these needed variable to np array


        sess = tf.InteractiveSession()
        tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()).run()
        self.queue_thread.reset(sess)
        path_source_indices_matrix, node_indices_matrix, path_target_indices_matrix, \
        source_word_embed_matrix, path_embed_matrix, target_word_embed_matrix, \
        np_target_string, self.source_string, self.path_string, self.target_string  = sess.run(
                [path_source_indices, node_indices, path_target_indices, 
                source_word_embed, path_embed, target_word_embed, target_string, 
                source_string, path_string, target_string])
        sess.close()

        self.path_source_indices_matrix, self.node_indices_matrix, self.path_target_indices_matrix \
            = np.array(path_source_indices_matrix), np.array(node_indices_matrix), np.array(path_target_indices_matrix)
        self.source_word_embed_matrix, self.path_embed_matrix, self.target_word_embed_matrix  = \
            np.array(source_word_embed_matrix), np.array(path_embed_matrix), np.array(target_word_embed_matrix)


        print("=============================================================\n")

        #print('Path source indices: ', self.path_source_indices_matrix)
        print('Path source indices: ', self.path_source_indices_matrix.shape)
        print("------------------------------\n")

        #print('Node indices: ', self.node_indices_matrix)
        print('Node indices: ', self.node_indices_matrix.shape)
        print("------------------------------\n")

        #print('Path target indices: ', self.path_target_indices_matrix)
        print('Path target indices: ', self.path_target_indices_matrix.shape)
        print("------------------------------\n")

        #print('source_word_embed: ', self.source_word_embed_matrix)
        print('source_word_embed: ', self.source_word_embed_matrix.shape)
        print("------------------------------\n")

        #print('path_embed: ', np.array(path_embed_temp))
        print('path_embed: ', self.path_embed_matrix.shape)
        print("------------------------------\n")
        
        #print('target_word_embed: ', np.array(target_word_embed_temp))
        print('target_word_embed: ', self.target_word_embed_matrix.shape)

        print("=============================================================\n")
        self.target_string = Common.binary_to_string_list(np_target_string)
        print('np_target_string: ', self.target_string)
        print('np_target_string shape: ', np_target_string.shape)

        #temp = self.calculate_total_distance(self.source_word_embed_matrix[0],
        #    self.path_embed_matrix[0],self.target_word_embed_matrix[0],
        #    self.source_word_embed_matrix[1],self.path_embed_matrix[1],self.target_word_embed_matrix[1])
        #print(temp)

        self.group_for_each_data = np.full(len(self.source_word_embed_matrix),0)

        #generate random K centers for clustering
        self.index_of_initial_center = self.generate_center(0,len(self.source_word_embed_matrix)-1)

        #self.sort_all_data()
        #self.update_center()



    def calculate_path_abstraction(self, path_embed, path_lengths, valid_contexts_mask, is_evaluating=False):
        return self.path_rnn_last_state(is_evaluating, path_embed, path_lengths, valid_contexts_mask)

    def path_rnn_last_state(self, is_evaluating, path_embed, path_lengths, valid_contexts_mask):
        # path_embed:           (batch, max_contexts, max_path_length+1, dim)
        # path_length:          (batch, max_contexts)
        # valid_contexts_mask:  (batch, max_contexts)
        max_contexts = tf.shape(path_embed)[1]
        flat_paths = tf.reshape(path_embed, shape=[-1, self.config.MAX_PATH_LENGTH,
                                                   self.config.EMBEDDINGS_SIZE])  # (batch * max_contexts, max_path_length+1, dim)
        flat_valid_contexts_mask = tf.reshape(valid_contexts_mask, [-1])  # (batch * max_contexts)
        lengths = tf.multiply(tf.reshape(path_lengths, [-1]),
                              tf.cast(flat_valid_contexts_mask, tf.int32))  # (batch * max_contexts)
        if self.config.BIRNN:
            rnn_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE / 2)
            rnn_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE / 2)
            if not is_evaluating:
                rnn_cell_fw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fw,
                                                            output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
                rnn_cell_bw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bw,
                                                            output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=rnn_cell_fw,
                cell_bw=rnn_cell_bw,
                inputs=flat_paths,
                dtype=tf.float32,
                sequence_length=lengths)
            final_rnn_state = tf.concat([state_fw.h, state_bw.h], axis=-1)  # (batch * max_contexts, rnn_size)  
        else:
            rnn_cell = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE)
            if not is_evaluating:
                rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
            _, state = tf.nn.dynamic_rnn(
                cell=rnn_cell,
                inputs=flat_paths,
                dtype=tf.float32,
                sequence_length=lengths
            )
            final_rnn_state = state.h  # (batch * max_contexts, rnn_size)

        return tf.reshape(final_rnn_state,
                          shape=[-1, max_contexts, self.config.RNN_SIZE])  # (batch, max_contexts, rnn_size)

    def compute_contexts(self, subtoken_vocab, nodes_vocab, source_input, nodes_input,
                         target_input, valid_mask, path_source_lengths, path_lengths, path_target_lengths,
                         is_evaluating=False):

        source_word_embed = tf.nn.embedding_lookup(params=subtoken_vocab,
                                                   ids=source_input)  # (batch, max_contexts, max_name_parts, dim)
        path_embed = tf.nn.embedding_lookup(params=nodes_vocab,
                                            ids=nodes_input)  # (batch, max_contexts, max_path_length+1, dim)
        target_word_embed = tf.nn.embedding_lookup(params=subtoken_vocab,
                                                   ids=target_input)  # (batch, max_contexts, max_name_parts, dim)

        source_word_mask = tf.expand_dims(
            tf.sequence_mask(path_source_lengths, maxlen=self.config.MAX_NAME_PARTS, dtype=tf.float32),
            -1)  # (batch, max_contexts, max_name_parts, 1)
        target_word_mask = tf.expand_dims(
            tf.sequence_mask(path_target_lengths, maxlen=self.config.MAX_NAME_PARTS, dtype=tf.float32),
            -1)  # (batch, max_contexts, max_name_parts, 1)

        source_words_sum = tf.reduce_sum(source_word_embed * source_word_mask,
                                         axis=2)  # (batch, max_contexts, dim)
        path_nodes_aggregation = self.calculate_path_abstraction(path_embed, path_lengths, valid_mask,
                                                                 is_evaluating)  # (batch, max_contexts, rnn_size)
        target_words_sum = tf.reduce_sum(target_word_embed * target_word_mask, axis=2)  # (batch, max_contexts, dim)

        context_embed = tf.concat([source_words_sum, path_nodes_aggregation, target_words_sum],
                                  axis=-1)  # (batch, max_contexts, dim * 2 + rnn_size)
        if not is_evaluating:
            context_embed = tf.nn.dropout(context_embed, self.config.EMBEDDINGS_DROPOUT_KEEP_PROB)

        batched_embed = tf.layers.dense(inputs=context_embed, units=self.config.DECODER_SIZE,
                                        activation=tf.nn.tanh, trainable=not is_evaluating, use_bias=False)


        return source_word_embed, path_embed, target_word_embed

    @staticmethod
    def initialize_session_variables(sess):
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()))


    def calculate_each_distance(self, input_vector, center):

        #Here, we have three embedding variables with dimension
        #self.source_word_embed_matrix, self.path_embed_matrix, self.target_word_embed_matrix
        #[number of methods, maximum contexts extracted from each method, 
        #    number of sub-tokens in contexts/sources/target, size of embedding vector for each word]
        #We input by each method and sources, paths or target
        #So the input dimension should be 
        #[maximum contexts extracted from each method, 
        #    number of sub-tokens in contexts/sources/target, size of embedding vector for each word]
        # calculate l2 norm for each of three to center

        return np.linalg.norm(input_vector-center)
        #return abs(np.sum(input_vector-center))

        MAX_NUM_OF_CONTEXTS = len(input_vector)
        NUM_OF_SUB_TOKEN = len(input_vector[0])
        SIZE_OF_EMBED_VECTOR = len(input_vector[0][0])

        average_distance = 0
        for i in range(MAX_NUM_OF_CONTEXTS):
            for j in range(NUM_OF_SUB_TOKEN):
                for k in range(SIZE_OF_EMBED_VECTOR):
                    average_distance = average_distance + math.sqrt(abs(input_vector[i][j][k]**2 - center[i][j][k]**2))

        average_distance = average_distance/(MAX_NUM_OF_CONTEXTS*NUM_OF_SUB_TOKEN*SIZE_OF_EMBED_VECTOR)
        #print(average_distance)

        return average_distance


    def calculate_total_distance(self,sample_source, sample_path, sample_target,
        center_source, center_path, center_target):

        total_distance = 0
        total_distance = total_distance + self.calculate_each_distance(sample_source,center_source)
        total_distance = total_distance + self.calculate_each_distance(sample_path,center_path)
        total_distance = total_distance + self.calculate_each_distance(sample_target,center_target)

        #print(total_distance)
        return total_distance

    def generate_center(self,lower_limit, upper_limit):
        #Here, we just randomly pick K centers from data-set
        index_list = []
        for i in range(0,self.K):
            n = random.randint(lower_limit,upper_limit)
            while n in index_list:
                n = random.randint(lower_limit,upper_limit)
            index_list.append(n)
            self.saved_center_source.append(self.source_word_embed_matrix[n])
            self.saved_center_path.append(self.path_embed_matrix[n])
            self.saved_center_target.append(self.target_word_embed_matrix[n])
        #print(index_list)
        print("=======Generated center==========\n")
        #print(self.saved_center_source)
        #print(self.saved_center_path)
        #print(self.saved_center_target)
        return index_list

    def sort_each_point(self, data_index):

        distance_array = []

        for i in range(0,self.K):
            cur_distance = self.calculate_total_distance(self.source_word_embed_matrix[data_index],
            self.path_embed_matrix[data_index],self.target_word_embed_matrix[data_index],
            self.source_word_embed_matrix[self.index_of_initial_center[i]],
            self.path_embed_matrix[self.index_of_initial_center[i]],
            self.target_word_embed_matrix[self.index_of_initial_center[i]])
            distance_array.append(cur_distance)

        #print(np.argmin(distance_array))
        #print(self.index_of_initial_center[np.argmin(distance_array)])

        self.group_for_each_data[data_index] = int(np.argmin(distance_array))
        #print(self.group_for_each_data)

    def sort_all_data(self):

        for i in range(len(self.source_word_embed_matrix)):
            self.sort_each_point(i)
            #print(self.group_for_each_data)

    def update_center(self):

        source_sum_of_groups = np.zeros(shape=np.array(self.saved_center_source).shape)
        path_sum_of_groups = np.zeros(shape=np.array(self.saved_center_path).shape)
        target_sum_of_groups = np.zeros(shape=np.array(self.saved_center_target).shape)
        length_of_each_group = np.full(self.K,0)
        #print(source_sum_of_groups,source_sum_of_groups.shape)

        for i in range(len(self.group_for_each_data)):
            source_sum_of_groups[self.group_for_each_data[i]] += self.source_word_embed_matrix[i]
            path_sum_of_groups[self.group_for_each_data[i]] += self.path_embed_matrix[i]
            target_sum_of_groups[self.group_for_each_data[i]] += self.target_word_embed_matrix[i]
            length_of_each_group += 1

        #print("=======Old center==========\n")
        #print(self.saved_center_source)
        #print(self.saved_center_path)
        #print(self.saved_center_target)

        for i in range(self.K):
            self.saved_center_source[i] = source_sum_of_groups[i]/length_of_each_group[i]
            self.saved_center_path[i] = path_sum_of_groups[i]/length_of_each_group[i]
            self.saved_center_target[i] = target_sum_of_groups[i]/length_of_each_group[i]


        #print("=======New center==========\n")
        #print(self.saved_center_source)
        #print(self.saved_center_path)
        #print(self.saved_center_target)

    def save_sorted_groups(self):
        np.savetxt('sorted_groups.txt', self.group_for_each_data, delimiter=' ')

    def load_sorted_groups(self):
        f = open("sorted_groups.txt", "rb")
        read_sorted_groups = np.array(np.loadtxt(f, delimiter=' '))
        #print(read_sorted_groups)
        #print(read_sorted_groups.shape)

    def save_centers(self, input_center, save_file_name):
        input_center_array = np.array(input_center)
        flatten_saved_center = input_center_array.flatten()
        for i in range(len(input_center_array.shape)):
            #print(np.array(input_center_array.shape)[len(input_center_array.shape)-1-i])
            flatten_saved_center = np.insert(flatten_saved_center, 0, 
                np.array(input_center_array.shape)[len(input_center_array.shape)-1-i])

        flatten_saved_center = np.insert(flatten_saved_center, 0, len(input_center_array.shape))

        np.savetxt(save_file_name, flatten_saved_center, delimiter=' ')

    def load_centers(self, save_file_name):
        f = open(save_file_name, "rb")
        read_source_center = np.array(np.loadtxt(f, delimiter=' '))

        len_of_matrix_shape = int(read_source_center[0])
        shape_of_matrix = read_source_center[1:len_of_matrix_shape+1].astype(int)
        contents_of_matrix = read_source_center[len_of_matrix_shape+1:]
        reshaped_matrix = contents_of_matrix.reshape(shape_of_matrix)
        #print(shape_of_matrix)
        #print(contents_of_matrix)
        #print(reshaped_matrix)
        #print(reshaped_matrix.shape)

    def save_all_centers(self):
        self.save_centers(self.saved_center_source, 'source_center.txt')
        self.save_centers(self.saved_center_path, 'path_center.txt')
        self.save_centers(self.saved_center_target, 'target_center.txt')

    def load_all_centers(self):
        self.load_centers('source_center.txt')
        self.load_centers('path_center.txt')
        self.load_centers('target_center.txt')

    def check_null_group(self):
        label = "null|dereference"
        for i in range(len(self.source_word_embed_matrix)):
            if self.target_string[i]==label:
                print(self.group_for_each_data[i])
                print(self.source_string[i])
                print(self.path_string[i])
                print(self.target_string[i])


    def cal_distance_for_null(self):
        label = "null|dereference"
        for i in range(len(self.source_word_embed_matrix)):
            if self.target_string[i]==label:
                dist = self.calculate_total_distance(self.source_word_embed_matrix[i],
                    self.path_embed_matrix[i],self.target_word_embed_matrix[i],
                    self.predict_source_word_embed_matrix[0],
                    self.predict_path_embed_matrix[0],self.predict_target_word_embed_matrix[0])
                print(dist)
                print(self.group_for_each_data[i])


    @staticmethod
    def read_file(input_filename):
        with open(input_filename, 'r') as file:
            return file.readlines()


    def get_the_processed_input(self, input_tensors):
        
        target_string = input_tensors[reader.TARGET_STRING_KEY]
        target_index = input_tensors[reader.TARGET_INDEX_KEY]
        target_lengths = input_tensors[reader.TARGET_LENGTH_KEY]
        path_source_indices = input_tensors[reader.PATH_SOURCE_INDICES_KEY]
        node_indices = input_tensors[reader.NODE_INDICES_KEY]
        path_target_indices = input_tensors[reader.PATH_TARGET_INDICES_KEY]
        valid_context_mask = input_tensors[reader.VALID_CONTEXT_MASK_KEY]
        path_source_lengths = input_tensors[reader.PATH_SOURCE_LENGTHS_KEY]
        path_lengths = input_tensors[reader.PATH_LENGTHS_KEY]
        path_target_lengths = input_tensors[reader.PATH_TARGET_LENGTHS_KEY]

        with tf.variable_scope('model'):


            source_word_embed = tf.nn.embedding_lookup(params=self.subtoken_vocab,
                                                       ids=path_source_indices)  # (batch, max_contexts, max_name_parts, dim)
            path_embed = tf.nn.embedding_lookup(params=self.nodes_vocab,
                                                ids=node_indices)  # (batch, max_contexts, max_path_length+1, dim)
            target_word_embed = tf.nn.embedding_lookup(params=self.subtoken_vocab,
                                                       ids=path_target_indices)  # (batch, max_contexts, max_name_parts, dim)



        #sess.close()
        sess = tf.InteractiveSession()
        tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()).run()
        self.predict_queue.reset(sess)
        path_source_indices_matrix, node_indices_matrix, path_target_indices_matrix, \
        source_word_embed_matrix, path_embed_matrix, target_word_embed_matrix, \
        np_target_string  = sess.run(
                [path_source_indices, node_indices, path_target_indices, 
                source_word_embed, path_embed, target_word_embed, target_string])


        self.predict_path_source_indices_matrix, self.predict_node_indices_matrix, self.predict_path_target_indices_matrix \
            = np.array(path_source_indices_matrix), np.array(node_indices_matrix), np.array(path_target_indices_matrix)
        self.predict_source_word_embed_matrix, self.predict_path_embed_matrix, self.predict_target_word_embed_matrix  = \
            np.array(source_word_embed_matrix), np.array(path_embed_matrix), np.array(target_word_embed_matrix)
        

        print("=============================================================\n")

        #print('Path source indices: ', self.predict_path_source_indices_matrix)
        print('Path source indices: ', self.predict_path_source_indices_matrix.shape)
        print("------------------------------\n")

        #print('Node indices: ', self.predict_node_indices_matrix)
        print('Node indices: ', self.predict_node_indices_matrix.shape)
        print("------------------------------\n")

        #print('Path target indices: ', self.path_target_indices_matrix)
        print('Path target indices: ', self.predict_path_target_indices_matrix.shape)
        print("------------------------------\n")

        #print('source_word_embed: ', self.predict_source_word_embed_matrix)
        print('source_word_embed: ', self.predict_source_word_embed_matrix.shape)
        print("------------------------------\n")

        #print('path_embed: ', np.array(path_embed_temp))
        print('path_embed: ', self.predict_path_embed_matrix.shape)
        print("------------------------------\n")
        
        #print('target_word_embed: ', np.array(target_word_embed_temp))
        print('target_word_embed: ', self.predict_target_word_embed_matrix.shape)

        print("=============================================================\n")
        print('np_target_string: ', Common.binary_to_string_list(np_target_string))
        print('np_target_string shape: ', np_target_string.shape)

        self.predict_group_for_each_data = np.full(len(self.predict_source_word_embed_matrix),0)

    def predict_sort_each_point(self, data_index):

        distance_array = []

        for i in range(0,self.K):
            cur_distance = self.calculate_total_distance(self.predict_source_word_embed_matrix[data_index],
            self.predict_path_embed_matrix[data_index],self.predict_target_word_embed_matrix[data_index],
            self.saved_center_source[i],
            self.saved_center_path[i],
            self.saved_center_target[i])
            distance_array.append(cur_distance)

        #print(np.argmin(distance_array))
        #print(self.index_of_initial_center[np.argmin(distance_array)])

        self.predict_group_for_each_data[data_index] = int(np.argmin(distance_array))
        #print(self.group_for_each_data)

    def predict_sort_all_data(self):

        for i in range(len(self.predict_source_word_embed_matrix)):
            self.predict_sort_each_point(i)
            #print(self.predict_group_for_each_data)

    def predict(self):

        SHOW_TOP_CONTEXTS = 10
        MAX_PATH_LENGTH = 8
        MAX_PATH_WIDTH = 2
        EXTRACTION_API = 'https://po3g2dx2qa.execute-api.us-east-1.amazonaws.com/production/extractmethods'
        exit_keywords = ['exit', 'quit', 'q']

        path_extractor = Extractor(self.config, EXTRACTION_API, self.config.MAX_PATH_LENGTH, max_path_width=2)


        input_filename = 'Input.java'
        print('Serving')

        #while True:
        #print('Modify the file: "' + input_filename + '" and press any key when ready, or "q" / "exit" to exit')
        #user_input = input()
        #if user_input.lower() in exit_keywords:
        #    print('Exiting...')
        #    return

        user_input = ' '.join(self.read_file(input_filename))
        try:
            predict_lines, pc_info_dict = path_extractor.extract_paths(user_input)
            #print(predict_lines)
            #print(pc_info_dict)
        except ValueError:
            pass


        if self.predict_queue is None:
            self.predict_queue = reader.Reader(subtoken_to_index=self.subtoken_to_index,
                                               node_to_index=self.node_to_index,
                                               target_to_index=self.target_to_index,
                                               config=self.config, is_evaluating=True)

            self.get_the_processed_input(self.predict_queue.get_output())


            self.predict_sort_all_data()
            print(self.predict_group_for_each_data)

            self.cal_distance_for_null()

            return

            self.predict_placeholder = tf.placeholder(tf.string)
            reader_output = self.predict_queue.process_from_placeholder(self.predict_placeholder)
            reader_output = {key: tf.expand_dims(tensor, 0) for key, tensor in reader_output.items()}
            

            self.predict_source_string = reader_output[reader.PATH_SOURCE_STRINGS_KEY]
            self.predict_path_string = reader_output[reader.PATH_STRINGS_KEY]
            self.predict_path_target_string = reader_output[reader.PATH_TARGET_STRINGS_KEY]
            self.predict_target_strings_op = reader_output[reader.TARGET_STRING_KEY]


            sess = tf.InteractiveSession()
            tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()).run()
            self.predict_queue.reset(sess)

            for line in predict_lines:
                true_target_strings, path_source_string, path_strings, path_target_string = self.sess.run(
                    [self.predict_target_strings_op, self.predict_source_string, 
                    self.predict_path_string, self.predict_path_target_string],
                    feed_dict={self.predict_placeholder: line})

                print(path_source_string.shape)
                print(path_strings.shape)
                print(path_target_string.shape)
                print(true_target_strings.shape)
        return






