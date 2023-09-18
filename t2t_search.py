import numpy as np


def get_similar_sentences(sentence_bert, transcript, target_sentence):
    # 긴 문장을 문장 단위로 분리
    sentences = transcript.split('.')
    

    transcript_embeddings = sentence_bert.encode(sentences)
    target_embedding = sentence_bert.encode(target_sentence)

    indices = get_most_similar_indices(target_embedding, transcript_embeddings)
    print("sentence rank indices : ", indices)
    return [[index, sentences[index]] for index in indices]
    

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_most_similar_indices(target_embedding, embeddings):
    similarities = []
    
    for i, embedding in enumerate(embeddings):
        similarity = cosine_similarity(target_embedding, embedding)
        similarities.append((i, similarity))
    
    # 유사도가 높은 순서대로 정렬
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    # index만 반환
    return [index for index, _ in sorted_similarities]


def get_summed_rank(image_ranks, image_times, image_names, sentence_ranks, seperated_transcript, response, num_displaying=3):
    '''
    input examples)
    image_ranks :  [10, 1, 0, 7, 6, 5, 2, 8, 9, 4, 3]
    sentence_ranks :  [[3, " He's not worth it, really"], [1, ' Just got worse'], [2, ' Hit me up'], [4, ''], [0, 'Company that gets killed puppies? Too horrible']]
    '''
    from speech2text import find_sentence_time
    top5_sentences = [i[1] for i in sentence_ranks[:5]]
    print("top5_sentences : ", top5_sentences)
    top5_sentences_times = []
    top5_sentences_temp = []
    for sentence in top5_sentences:
        st, et = find_sentence_time(response, sentence)
        if st == None:
            continue
        else:
            top5_sentences_times.append((st+et)/2)
            top5_sentences_temp.append(sentence)
    top5_sentences = top5_sentences_temp

    sentence2image_indices = find_closest_indices(image_times.values(), top5_sentences_times)

    
    summed_scores = []
    for rank, image_idx in enumerate(image_ranks):
        if image_idx in sentence2image_indices:
            # sentence2image 랭크가 높을수록 적은 패널티
            temp = [image_idx, rank+sentence2image_indices.index(image_idx)]
        else:
            # 패널티
            temp = [image_idx, rank+len(sentence2image_indices)]
        summed_scores.append(temp)
    summed_scores = sorted(summed_scores, key=lambda x:x[1])
    summed_ranks = [i[0] for i in summed_scores]

    image_top_rank = image_ranks[0]
    sentence_top_rank = sentence_ranks[0]

    print("image_ranks : ", image_ranks)
    print("sentence_ranks : ", sentence_ranks)
    print("summed_ranks : ", summed_ranks)

    ranks = [summed_ranks[0], image_ranks[0], sentence_ranks[0][0]]
    ranks = list(set(ranks))
    i = 0
    while len(ranks) < num_displaying:
        i += 1
        ranks.append(summed_ranks[i])
        ranks = list(set(ranks))

    
    print("final ranks : ", ranks)
    # print("image_times : ", image_times.values())
    # print("top5_sentences_times : ", top5_sentences_times)
    return ranks[:num_displaying]


def find_closest_indices(list1, list2):
    '''
    examle)
    list1 = [1,2,3,4,5,6,7,12,15,16]
    list2 = [3, 5.5, 13.5]
    >>> [2, 4, 7]
    '''
    result = []
    for num2 in list2:
        closest_index = min(range(len(list1)), key=lambda i: abs(list1[i] - num2))
        result.append(closest_index)
    return result






