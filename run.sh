# python normal_predict.py --dataset news_articles --use_best_hit
# python normal_predict.py --dataset news_articles --use_best_hit --model Qwen2.5-72B
# python normal_predict.py --dataset news_articles --use_best_hit --model deepseek-r1
# python hybrid_predict.py --dataset news_articles --use_best_hit

# python normal_predict.py --dataset news_articles --use_best_hit --model deepseek-r1
# python normal_predict.py --dataset 2wikimultihopqa --use_best_hit --model deepseek-r1
# python normal_predict.py --dataset musique --use_best_hit --model deepseek-r1
# python normal_predict.py --dataset hotpotqa --use_best_hit --model deepseek-r1

# python normal_predict.py --dataset news_articles
# python normal_predict.py --dataset news_articles --model Qwen2.5-72B
# python normal_predict.py --dataset musique --use_best_hit --model Qwen2.5-0.5B
# python hybrid_predict.py --dataset news_articles

python normal_predict.py --dataset 2wikimultihopqa
python normal_predict.py --dataset musique

python normal_predict.py --dataset hotpotqa --embed_model retrieve_fine_tuning/models/hotpotqa_bge-embedder-ft --rerank_model retrieve_fine_tuning/models/hotpotqa_bge-reranker-ft
python normal_predict.py --dataset 2wikimultihopqa --embed_model retrieve_fine_tuning/models/2wikimultihopqa_bge-embedder-ft --rerank_model retrieve_fine_tuning/models/2wikimultihopqa_bge-reranker-ft
python normal_predict.py --dataset musique --embed_model retrieve_fine_tuning/models/musique_bge-embedder-ft --rerank_model retrieve_fine_tuning/models/musique_bge-reranker-ft

python normal_predict.py --dataset hotpotqa --model Qwen2.5-72B --embed_model retrieve_fine_tuning/models/hotpotqa_bge-embedder-ft --rerank_model retrieve_fine_tuning/models/hotpotqa_bge-reranker-ft
python normal_predict.py --dataset 2wikimultihopqa --model Qwen2.5-72B --embed_model retrieve_fine_tuning/models/2wikimultihopqa_bge-embedder-ft --rerank_model retrieve_fine_tuning/models/2wikimultihopqa_bge-reranker-ft
python normal_predict.py --dataset musique --model Qwen2.5-72B --embed_model retrieve_fine_tuning/models/musique_bge-embedder-ft --rerank_model retrieve_fine_tuning/models/musique_bge-reranker-ft

python hybrid_predict_separate.py --dataset hotpotqa --embed_model retrieve_fine_tuning/models/hotpotqa_bge-embedder-ft --rerank_model retrieve_fine_tuning/models/hotpotqa_bge-reranker-ft
python hybrid_predict_separate.py --dataset musique --embed_model retrieve_fine_tuning/models/hotpotqa_bge-embedder-ft --rerank_model retrieve_fine_tuning/models/hotpotqa_bge-reranker-ft
python hybrid_predict_separate.py --dataset 2wikimultihopqa --embed_model retrieve_fine_tuning/models/hotpotqa_bge-embedder-ft --rerank_model retrieve_fine_tuning/models/hotpotqa_bge-reranker-ft

# lora fine-tuning
# python fine_tuning/fine_tuning.py --dataset musique
python normal_predict.py --dataset hotpotqa --use_best_hit --model Qwen2.5-7B
python hybrid_predict.py --dataset hotpotqa --use_best_hit --small_model Qwen2.5-7B
# python hybrid_predict.py --dataset hotpotqa --use_best_hit --small_model Qwen2.5-7B-ft

# python fine_tuning/fine_tuning.py --dataset 2wikimultihopqa
# python normal_predict.py --dataset 2wikimultihopqa --use_best_hit --model Qwen2.5-0.5B-ft

# train embedders
# python retrieve_fine_tuning/embed_fine_tuning_preprocess.py --dataset musique
# bash retrieve_fine_tuning/embed_fine_tuning.sh
# python retrieve_fine_tuning/embed_fine_tuning_preprocess.py --dataset hotpotqa
# bash retrieve_fine_tuning/embed_fine_tuning.sh
# python retrieve_fine_tuning/embed_fine_tuning_preprocess.py --dataset 2wikimultihopqa
# bash retrieve_fine_tuning/embed_fine_tuning.sh

# train rerankers
# python retrieve_fine_tuning/rerank_fine_tuning_preprocess.py --dataset musique
# bash retrieve_fine_tuning/rerank_fine_tuning.sh
# python retrieve_fine_tuning/rerank_fine_tuning_preprocess.py --dataset hotpotqa
# bash retrieve_fine_tuning/rerank_fine_tuning.sh
# python retrieve_fine_tuning/rerank_fine_tuning_preprocess.py --dataset 2wikimultihopqa
# bash retrieve_fine_tuning/rerank_fine_tuning.sh

# seperate
python normal_predict_separate.py --dataset musique --use_best_hit --model Qwen2.5-7B
python hybrid_predict_separate.py --dataset musique --use_best_hit

python hybrid_predict_separate_with_fasttext.py --dataset musique --use_best_hit

# reflection
python hybrid_predict_reflection.py --dataset musique --use_best_hit
python normal_predict_ref.py --dataset hotpotqa --use_best_hit --model Qwen2.5-7B

python hybrid_predict_separate_ref.py --dataset musique --use_best_hit