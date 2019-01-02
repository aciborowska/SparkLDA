from pyspark import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer, StopWordsRemover
import numpy as np
import math
import random
import time
import pyLDAvis

np.random.seed(1234)
random.seed(1234)

c_k = None
c_k_n = None
V = None

#Parameters
alpha = 0.5
beta = 0.2
K = 5
iterations = 1000
iterations_test = 200
vocab_size = 182
minDF = 0
param_update = 5
eval_every = 100
partitions_no = 10
training_ratio = 0.9
dataset_dir = 'datasets/'
dataset_file = 'abcnews-date-text_short.csv'
visualization = True
visualization_output_dir = 'results/vis/'
results_dir = "results/"
results = []

def init(row):
    z_doc = []
    c_k_doc = np.zeros(K)
    count = 0
    for idx, word_id in enumerate(row[1].indices):
        word_count = row[1].values[idx]
        topic = np.random.randint(0, K)
        z_doc.append((word_id, topic, word_count))
        c_k_doc[topic] += word_count
        count += word_count
    return row[0], z_doc, c_k_doc, count


def update_c_k_n(d):
    c = np.zeros((K, V), dtype=int)
    for item in d:
        word_id = int(item[0].split('_')[0])
        topic = int(item[0].split('_')[1])
        count = int(item[1])
        c[topic, word_id] += count
    return c


def update_c_k(d):
    c_n = np.zeros(K, dtype=int)
    for item in d:
        topic = int(item[0].split('_')[1])
        count = int(item[1])
        c_n[topic] += count
    return c_n


def get_c_k_m_x(z):
    matrix = np.zeros((len(z), K))
    for i in range(0, len(z)):
        matrix[i] = z[i][1]
    return matrix


def gibbs_sampling(z_m):
    c_k_m = z_m[2]
    c_k_local = c_k.value.copy()
    c_k_n_local = c_k_n.value.copy()

    for iteration in range(0, param_update):
        for idx, word_topic in enumerate(z_m[1]):
            word_id = word_topic[0]
            topic = word_topic[1]
            count = word_topic[2]
            c_k_local[topic] -= count
            c_k_n_local[topic][word_id] -= count
            c_k_m[topic] -= count

            p_z = np.zeros(K)
            for k in range(0, K):
                p_z[k] = ((c_k_m[k] + alpha) / (z_m[3] - count + (K * alpha))) * \
                         ((c_k_n_local[k][word_id] + beta) / (c_k_local[k] + (beta * V)))
            new_topic = np.random.multinomial(1, p_z / p_z.sum()).argmax()

            z_m[1][idx] = (word_id, new_topic, count)
            c_k_local[new_topic] += count
            c_k_n_local[new_topic][word_id] += count
            c_k_m[new_topic] += count

    return z_m[0], z_m[1], c_k_m, z_m[3]


def process_testset(z_m):
    c_k_m = z_m[2]
    c_k_local = c_k.value.copy()
    c_k_n_local = c_k_n.value.copy()

    for iteration in range(0, iterations_test):
        for idx, word_topic in enumerate(z_m[1]):
            word_id = word_topic[0]
            topic = word_topic[1]
            count = word_topic[2]
            c_k_m[topic] -= count

            p_z = np.zeros(K)
            for k in range(0, K):
                p_z[k] = ((c_k_m[k] + alpha) / (z_m[3] - count + (K * alpha))) * \
                         ((c_k_n_local[k][word_id] + beta) / (c_k_local[k] + (beta * V)))
            new_topic = np.random.multinomial(1, p_z / p_z.sum()).argmax()

            z_m[1][idx] = (word_id, new_topic, count)
            c_k_m[new_topic] += count

    return z_m[0], z_m[1], c_k_m, z_m[3]


def word_topics(z_m):
    wtt = []
    for word_topic in z_m[1]:
        word_id = word_topic[0]
        topic = word_topic[1]
        count = word_topic[2]
        wtt.append((str(word_id)+'_'+str(topic), count))
    return wtt


def compute_phi(c_k_x_n):
    phi = np.zeros((K, V))
    for k in range(0, K):
        for v in range(0, V):
            phi[k, v] = (c_k_x_n[k, v] + beta) / ((V * beta) + np.sum(c_k_x_n[k, :]))
    return phi


def compute_theta(c_k_m_x):
    theta = np.zeros((len(c_k_m_x), K))
    for m in range(0, len(c_k_m_x)):
        for k in range(0, K):
            theta[m, k] = (c_k_m_x[m, k] + alpha) / ((K * alpha) + np.sum(c_k_m_x[m, :]))
    return theta


def perplexity(docs, theta, phi):
    sum_nom = 0
    sum_docs_len = 0
    for d in docs:
        doc_id = d[0]
        sum_docs_len += np.sum(d[1].values)
        for word_id in d[1].indices:
            sum_nom -= np.log(np.inner(phi[:, word_id], theta[doc_id]))

    return math.exp(sum_nom / sum_docs_len)


def word_count(row):
    counts = []
    for idx in range(0, len(row[1].indices)):
        word_id = row[1].indices[idx]
        count = row[1].values[idx]
        counts.append((word_id, count))
    return counts


start_time = time.time()
with SparkSession.builder \
        .master("local[*]") \
        .appName("LDA") \
        .getOrCreate() as spark:

    sc = spark.sparkContext
    sqlContext = SQLContext(sc)

    log4jLogger = sc._jvm.org.apache.log4j
    log = log4jLogger.LogManager.getLogger(__name__)
    log.warn("Spark starting...")

    # 1. Read and process data
    log.warn("Processing dataset")
    textFile = sc.textFile(dataset_dir+dataset_file, minPartitions=partitions_no)
    rdd = textFile.map(lambda line: line.split(',')[1], preservesPartitioning=True)\
                  .map(lambda doc: doc.split(' '), preservesPartitioning=True)\
                  .map(lambda word: [x for x in word if len(x) > 2], preservesPartitioning=True) \
                  .map(lambda word: [x.lower() for x in word], preservesPartitioning=True) \
                  .zipWithIndex().cache()

    df = sqlContext.createDataFrame(rdd, ['text', 'id'])
    M = df.count()
    log.warn("Number of docs = {0}".format(M))

    # Remove stop words
    remover = StopWordsRemover(inputCol="text", outputCol="filtered")
    df_filtered = remover.transform(df).select('id', 'filtered')

    # Divide for testing and training datasets
    training_end_idx = int(training_ratio * M)
    training_set_raw = df_filtered.filter(df_filtered.id < training_end_idx).repartition(partitions_no)
    testing_set_raw = df_filtered.filter(df_filtered.id >= training_end_idx).repartition(partitions_no)

    # 2. Create vocabulary
    log.warn("Building vocabulary")
    cv_model = CountVectorizer(inputCol="filtered", outputCol="vectors", minDF=minDF, vocabSize=vocab_size).fit(training_set_raw)
    V = len(cv_model.vocabulary)
    log.warn("Vocabulary size = {0}".format(V))

    # 3. Transform documents to BOW representation:
    # each doc is represented as SparseVector: (vocabSize, {word_id:count, word_id:count,...}
    log.warn("Transform training dataset to bow representation")
    training_set = cv_model.transform(training_set_raw).select('id', 'vectors').cache()
    log.warn('Training set: {0} documents'.format(training_set.count()))
    training_set_local = training_set.collect()

    # 4. Initialize model:
    # 4.1 each doc represented by (id, z_n array (topic to word assignment) and c_k_m (topics distribution for doc)
    # 4.2 randomly assign topic to each word in document, increment c_k_m accordingly
    z_m_n = training_set.rdd.map(init, preservesPartitioning=True).cache()

    z_m_n_matrix = z_m_n.flatMap(word_topics).reduceByKey(lambda a, b: a + b).collect()
    c_k_global = update_c_k(z_m_n_matrix)
    c_k_n_global = update_c_k_n(z_m_n_matrix)

    c_k_m_x = get_c_k_m_x(z_m_n.map(lambda x: (x[0], x[2])).sortByKey(ascending=True).collect())
    theta = compute_theta(c_k_m_x)
    phi = compute_phi(c_k_n_global)
    perplex = perplexity(training_set_local, theta, phi)
    results.append(('init train', 0, perplex))
    log.warn('Initial perplexity = {0}'.format(perplex))

    # 5. LDA with collapsed gibbs sampling
    log.warn("Training...")
    for i in range(0, iterations, param_update):

        c_k = sc.broadcast(c_k_global)
        c_k_n = sc.broadcast(c_k_n_global)

        new_z = z_m_n.map(gibbs_sampling, preservesPartitioning=True).cache()

        z_m_n_matrix = new_z.flatMap(word_topics).reduceByKey(lambda a, b: a+b).collect()
        c_k_global = update_c_k(z_m_n_matrix)
        c_k_n_global = update_c_k_n(z_m_n_matrix)

        if eval_every is not None and i > 0 and ((i >= 100 and i % eval_every == 0) or (i < 100 and i % 5 == 0)):
            c_k_m_x = get_c_k_m_x(new_z.map(lambda x: (x[0], x[2])).sortByKey(ascending=True).collect())
            theta = compute_theta(c_k_m_x)
            phi = compute_phi(c_k_n_global)
            perplex = perplexity(training_set_local, theta, phi)
            results.append(('train', i, perplex))
            log.warn('Iteration {0} - perplexity = {1}'.format(i, perplex))

        z_m_n = None
        z_m_n = new_z
        new_z = None

    # 6. Calculate phi and theta
    c_k_m_x = get_c_k_m_x(z_m_n.map(lambda x: (x[0], x[2]), preservesPartitioning=True).sortByKey(ascending=True).collect())
    theta = compute_theta(c_k_m_x)
    phi = compute_phi(c_k_n_global)
    perplex = perplexity(training_set_local, theta, phi)
    results.append(('train', iterations, perplex))
    log.warn('Final train set perplexity = {0}'.format(perplex))

    # 7. Evaluate perplexity on the testing set
    log.warn("Transform test dataset to bow representation")
    testing_set = cv_model.transform(testing_set_raw).select('id', 'vectors').rdd.map(lambda x: (x[0]-450, x[1])).repartition(partitions_no)
    testing_set_local = testing_set.collect()
    log.warn('Test set: {0} documents'.format(len(testing_set_local)))

    z_m_n_test = testing_set.map(init, preservesPartitioning=True).cache()
    c_k_m_x = get_c_k_m_x(z_m_n_test .map(lambda x: (x[0], x[2])).sortByKey(ascending=True).collect())
    theta_test = compute_theta(c_k_m_x)
    perplex = perplexity(testing_set_local, theta_test, phi)
    results.append(('test', 0, perplex))
    log.warn('Initial test set perplexity = {1}'.format(i, perplex))

    new_z_test = z_m_n_test.map(process_testset, preservesPartitioning=True).cache()

    c_k_m_x = get_c_k_m_x(new_z_test.map(lambda x: (x[0], x[2]), preservesPartitioning=True)
                          .sortByKey(ascending=True).collect())
    theta_test = compute_theta(c_k_m_x)
    perplex = perplexity(testing_set_local, theta_test, phi)
    results.append(('test', iterations_test, perplex))
    log.warn('Testing set perplexity = {0}'.format(perplex))

    # 8. Print words in topic distribution
    for topic in range(0, K):
        log.warn("Topic {0}".format(topic))
        word_ids = np.argpartition(phi[topic], -4)[-4:]
        for word_id in word_ids:
            log.warn('{0}: {1}'.format(cv_model.vocabulary[word_id], phi[topic, word_id]))

    end_time = time.time()
    log.warn("Execution time = {0} s".format(end_time-start_time))
    results.append(('time', end_time - start_time, 0))

    # 9. Save perplexity and time to file:
    filename = 'result_dataset={0}_k={1}_V={2}_update={3}_iter={4}.{5}'
    if len(results) > 0:
        with open(results_dir+filename.format(dataset_file,K,V,param_update,iterations,'csv'), 'w') as f:
            for result in results:
                f.write('{0},{1},{2}\n'.format(result[0], result[1], result[2]))


    # 9. Create and save visualization to file
    if visualization:
        docs_len = []
        for doc in training_set_local:
            docs_len.append(np.sum(doc[1].values))

        word_frequency = training_set.rdd.flatMap(word_count).reduceByKey(lambda a, b: a + b).sortByKey().values().collect()

        plot = pyLDAvis.prepare(phi, theta, docs_len, cv_model.vocabulary, word_frequency)
        pyLDAvis.save_html(plot, visualization_output_dir+
                           filename.format(dataset_file,K,V,param_update,iterations, '.html'))

