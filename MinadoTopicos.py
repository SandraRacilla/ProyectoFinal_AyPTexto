"""
#!/usr/bin/python
# Fecha:            07/Febrero/2021
# Descripción:      Programa que permite realizar el minado de tópicos
"""
import pandas as pd
import numpy as np
import os
##Uso de TFIDF
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import random

##Uso de LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def txt_NMF(reviews_datasets_NMF):
    """
    Antes aplicar el algoritmo de NMF es necesario obtener el vocabulario del archivo
    NMF hace uso de TFIDF
    max_df=0.80 -> Palabras que aparezcan al menos en 80% del documento
    min_df=2 -> Palabras que aparezcan al menos en 2 documentos
    """
    my_stop_words=text.ENGLISH_STOP_WORDS.union(["https"],["nhttps"],["d4leu57x7h"])
    tfidf_vect = TfidfVectorizer(max_df=0.8, min_df=2, stop_words=my_stop_words)
    ##Matriz generada con TFIDF
    doc_term_matrix = tfidf_vect.fit_transform(reviews_datasets_NMF['data__text'].values.astype('U'))
    print(reviews_datasets_NMF['data__text'].values.astype('U'))
    """
    Uso de NMF para crear temas junto con la distribución de probabilidad para cada palabra del vocabulario
    n_components:5 -> Numero de categorias o temas que en las que queremos
                        que se divida nuestro texto
    random_state:42 -> seed 
    Creamos una matriz de probabilidad con las probabilidades de todas las palabras en el vocabulario
    """
    nmf = NMF(n_components=5, random_state=42)
    nmf.fit(doc_term_matrix )

    #print(len(tfidf_vect.get_feature_names()))
    print(tfidf_vect.get_feature_names())
    """
    ###Palabras de nuestro vocabulario
    for i in range(10):
        random_id = random.randint(0,len(tfidf_vect.get_feature_names()))
        print(tfidf_vect.get_feature_names()[random_id])
    """
    ##Para encontrar el primer topic se usa "components_" con atributo 0
    first_topic = nmf.components_[0]
    ##first_topic contiene la probabilidad de 3716 palabras para el topic 1
    ##Ordenamos los índices de acuerdo a los valores de las probabilidades
    ##Regresa indíces de 10 palabras con las probabilidades más altas
    top_topic_words = first_topic.argsort()[-10:]
    """
    Pasamos índices al vector para observar las palabras
    
    for i in top_topic_words:
        print(tfidf_vect.get_feature_names()[i])
    """
    fic = open("topics_NMF.txt", "w")
    print('\t\t\t\tTemas NMF', file=fic)
    for i,topic in enumerate(nmf.components_):
        print(f'NMF Top 10 words for topic #{i}:', file=fic)
        print([tfidf_vect.get_feature_names()[i] for i in topic.argsort()[-10:]], file=fic)
        print('\n\n', file=fic)
    fic.close()

    topic_values = nmf.transform(doc_term_matrix)
    reviews_datasets_NMF['Topic'] = topic_values.argmax(axis=1)
    print(reviews_datasets_NMF.head())

def txt_LDA(reviews_datasets_LDA):
    """
    Antes aplicar el algoritmo de LDA es necesario obtener el vocabulario
    del archivo
    max_df=0.80 -> Palabras que aparezcan al menos en 80% del documento
    min_df=2 -> Palabras que aparezcan al menos en 2 documentos
    """
    my_stop_words=text.ENGLISH_STOP_WORDS.union(["https"],["nhttps"],["d4leu57x7h"])
    count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words=my_stop_words)
    doc_term_matrix = count_vect.fit_transform(reviews_datasets_LDA['data__text'].values.astype('U'))

    """
    Uso de LDA para crear temas junto con la distribución de probabilidad 
    para cada palabra del vocabulario
    n_components:5 -> Numero de categorias o temas que en las que queremos
                        que se divida nuestro texto
    random_state:42 -> seed 
    """
    LDA = LatentDirichletAllocation(n_components=5, random_state=42)
    LDA.fit(doc_term_matrix)
    """
    for i in range(10):
        random_id = random.randint(0,len(count_vect.get_feature_names()))
        print(count_vect.get_feature_names()[random_id])
    """

    """
    Encontramos 10 palabras con la probabilidad más alta para los temas
    first_topic contiene las probabilidades de 3716 palabras para el tema 1
    argsort() Ordenar índices de acuerdo a los valores de probabilidades
    [-10:] Toma los últimos 10 valores, es decir los que tienen mayor valor  
    """
    first_topic = LDA.components_[0]
    #print(len(first_topic))
    top_topic_words = first_topic.argsort()[-10:]
    #print(top_topic_words)

    """
    Obtenemos palabras relacionadas con los índices anteriores
    
    for i in top_topic_words:
        print(count_vect.get_feature_names()[i])
    """
    
    """
    Impresion de 10 palabras con mayor probabilidad de cada uno de los 5 temas 
    """
    fic = open("topics_LDA.txt", "w")
    print('\t\t\t\tTemas LDA', file=fic)
    for i,topic in enumerate(LDA.components_):
        print(f'LDA Top 10 words for topic #{i}:', file=fic)
        print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]], file=fic)
        print('\n\n', file=fic)
    fic.close()

    """
    Agregamos una columna al archivo donde agreguemos el tema al que pertenece
    """
    topic_values = LDA.transform(doc_term_matrix)
    topic_values.shape
    reviews_datasets_LDA['Topic'] = topic_values.argmax(axis=1)
    print(reviews_datasets_LDA.head())


reviews_datasets = pd.read_csv(r'Tweets Recabados.csv')
reviews_datasets = reviews_datasets.head(2429)
reviews_datasets.dropna()

txt_LDA(reviews_datasets)
print('\n\n\n\n')
txt_NMF(reviews_datasets)