import json


def preprocess_news_articles():
    with open("/home/gpt/hgx01_share/data/news_articles/MultiHopRAG.json") as f:
        questions = json.load(f)
    with open("/home/gpt/hgx01_share/data/news_articles/corpus.json") as f:
        corpus = json.load(f)
    
    with open("data/news_articles/dev.json","w") as f:
        json.dump(questions, f, indent=4)
    with open("data/news_articles/corpus.json","w") as f:
        json.dump(corpus, f, indent=4)

if __name__ == '__main__':
    preprocess_news_articles()