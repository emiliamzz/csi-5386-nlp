import cohere
import nltk
from numpy import corrcoef
from sentence_transformers import SentenceTransformer


def score_and_write(embeddings, f):
    scoress = corrcoef(embeddings)
    for scores in scoress:
        for score in scores:
            f.write(str(score) + '\n')


if __name__ == '__main__':
    # Chose at least 5 pre-trained sentence embeddings. If they have different parameters, you can experiment with them,
    # but choose one for your report. Also make sure to include a version of SBERT (Reimers and Gurevych, 2019,
    # https://aclanthology.org/D19-1410/) (https://www.sbert.net/) and at least one model based on recent generative LLM
    # models, in addition to other pre-trained language models that can represent sentences.
    sbert = SentenceTransformer('all-mpnet-base-v2')
    co = cohere.Client('SECRET')

    # Use the dataset from the Semeval 2016-Task1 Semantic Textual Similarity (STS).
    # Use the test data STS Core (English Monolingual subtask) - test data with gold labels. Do not use the training
    # data. Read more about the task at https://alt.qcri.org/semeval2016/task1/#
    with open('./sts2016-english-with-gs-v1.0/STS2016.input.answer-answer.txt') as file:
        answer_answer_sentences = nltk.sent_tokenize(file.read())
    with open('./sts2016-english-with-gs-v1.0/STS2016.input.headlines.txt') as file:
        headline_sentences = nltk.sent_tokenize(file.read())
    with open('./sts2016-english-with-gs-v1.0/STS2016.input.plagiarism.txt') as file:
        plagiarism_sentences = nltk.sent_tokenize(file.read())
    with open('./sts2016-english-with-gs-v1.0/STS2016.input.postediting.txt') as file:
        postediting_sentences = nltk.sent_tokenize(file.read())
    with open('./sts2016-english-with-gs-v1.0/STS2016.input.question-question.txt') as file:
        question_question_sentences = nltk.sent_tokenize(file.read())

    # The evaluation score to report is the Pearson correlation between the score obtained by your model and the
    # expected solution. The expected solution scores are numeric values between and 5 (from low similarity to high
    # similarity).
    sbert_answer_answer_embeddings = sbert.encode(answer_answer_sentences)
    sbert_headline_embeddings = sbert.encode(headline_sentences)
    sbert_plagiarism_embeddings = sbert.encode(plagiarism_sentences)
    sbert_postediting_embeddings = sbert.encode(postediting_sentences)
    sbert_question_question_embeddings = sbert.encode(question_question_sentences)

    with open('./sbert/answer_answer.txt', 'w') as file:
        score_and_write(sbert_answer_answer_embeddings, file)
    with open('./sbert/headline.txt', 'w') as file:
        score_and_write(sbert_headline_embeddings, file)
    with open('./sbert/plagiarism.txt', 'w') as file:
        score_and_write(sbert_plagiarism_embeddings, file)
    with open('./sbert/postediting.txt', 'w') as file:
        score_and_write(sbert_postediting_embeddings, file)
    with open('./sbert/question_question.txt', 'w') as file:
        score_and_write(sbert_question_question_embeddings, file)

    co_answer_answer_embeddings = co.embed(
        texts=answer_answer_sentences,
        model='embed-english-v3.0',
        input_type='clustering',
        truncate='NONE'
    ).embeddings
    co_headline_embeddings = co.embed(
        texts=headline_sentences,
        model='embed-english-v3.0',
        input_type='clustering',
        truncate='NONE'
    ).embeddings
    co_plagiarism_embeddings = co.embed(
        texts=plagiarism_sentences,
        model='embed-english-v3.0',
        input_type='clustering',
        truncate='NONE'
    ).embeddings
    co_postediting_embeddings = co.embed(
        texts=postediting_sentences,
        model='embed-english-v3.0',
        input_type='clustering',
        truncate='NONE'
    ).embeddings
    co_question_question_embeddings = co.embed(
        texts=question_question_sentences,
        model='embed-english-v3.0',
        input_type='clustering',
        truncate='NONE'
    ).embeddings

    with open('./co/answer_answer.txt', 'w') as file:
        score_and_write(co_answer_answer_embeddings, file)
    with open('./co/headline.txt', 'w') as file:
        score_and_write(co_headline_embeddings, file)
    with open('./co/plagiarism.txt', 'w') as file:
        score_and_write(co_plagiarism_embeddings, file)
    with open('./co/postediting.txt', 'w') as file:
        score_and_write(co_postediting_embeddings, file)
    with open('./co/question_question.txt', 'w') as file:
        score_and_write(co_question_question_embeddings, file)
