from openai import OpenAI
from tqdm import tqdm
import json
import re
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers import ContextualCompressionRetriever, BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import FAISS
from BCERerank import BCERerank

def parse_answer(response):
    try:
        return response[re.search("Answer:\n", response).end():]
    except:
        try: 
            return response[re.search("answer:\n", response).end():]
        except:
            try:
                return response[re.search("Answer: ", response).end():]
            except:
                try:
                    return response[re.search("answer: ", response).end():]
                except:
                    try:
                        return response[re.search("Answer:", response).end():]
                    except:
                        try:
                            return response[re.search("answer:", response).end():]
                        except:
                            line = response.split("\n")[-1]
                            try:
                                return line[re.search(": ", line).end():]
                            except:
                                return line

client = OpenAI(
    api_key="vllm",
    base_url="http://localhost:6008/v1",
)
model_name = "llama3.1-8b-instruct"

client2 = OpenAI(
    api_key="vllm",
    base_url="http://localhost:6009/v1",
)
model_name2 = "llama3.1-70b-instruct"

question = "Who founded the company that distributed the film UHF?"
text = "\n".join([
            "In 2003, John Kirtland co-founded Kirtland Records with his wife, The Polyphonic Spree choir member Jenny Kirtland.",
            "Amblin Entertainment is an American film production company founded by director and producer Steven Spielberg, and film producers Kathleen Kennedy and Frank Marshall in 1981.",
            "Morris Mike Medavoy (born January 21, 1941) is an American film producer and executive, co-founder of Orion Pictures (1978), former chairman of TriStar Pictures, former head of production for United Artists (1974–1978) and current chairman and CEO of Phoenix Pictures.",
            "SModcast Pictures is an American film distribution company and a film and television production company founded by Kevin Smith in 2011.",
            "Kirtland Records went on to become the record label for a number of Texas-based indie bands, including the Burden Brothers, The Polyphonic Spree, Toadies, The Vanished, and Sarah Jaffe.",
            # "The company's headquarters are located on the backlot of Universal Studios in Universal City, California.",
            # "Founded in 1989, it is run by Bill Barbot and Kim Coletta, both formerly of the band Jawbox.",
            # "SICRAL 1B is a military communications satellite built by Thales Alenia Space for Italian Armed Forces.",
            # "The label is mainly distributed in Europe by Sony Music/EMI.",
            # "Videodrome is a 1983 Canadian science fiction body horror film written and directed by David Cronenberg and starring James Woods, Sonja Smits, and Deborah Harry."
        ])
question2 = "Where is Ulrich Walter's employer headquartered?"
text2 = "\n".join([
            "Its main offices are located at 30 Rockefeller Plaza at Rockefeller Center in New York City, known now as the Comcast Building.",
            "Simultaneously, the growth and expansion of Yale University further affected the economic shift.",
            "These include SunTrust Bank (based in Atlanta), Capital One Financial Corporation (officially based in McLean, Virginia, but founded in Richmond with its operations center and most employees in the Richmond area), and the medical and pharmaceutical giant McKesson (based in San Francisco).",
            "GE is a multinational conglomerate headquartered in Fairfield, Connecticut.",
            "After two post-doc positions at the Argonne National Laboratory, Chicago, Illinois, and the University of California at Berkeley, California, he was selected in 1987 to join the German astronaut team.",
            "From 1988 to 1990, he completed basic training at the German Aerospace Center, and was then nominated to be in the prime crew for the second German Spacelab mission.",
            "The John Deere World Headquarters is a complex of four buildings located on 1,400 acres (5.7 km²) of land at One John Deere Place, Moline, Illinois, United States.",
            "UPS Freight, the less-than-truckload division of UPS and formerly known as Overnite Transportation, has its corporate headquarters in Richmond.",
            "Schlage was headquartered in San Francisco from its inception until it relocated to Colorado Springs, Colorado in 1997.",
            "DuPont maintains a production facility in South Richmond known as the Spruance Plant."
        ])

embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': 'cuda:0'},
    encode_kwargs={'normalize_embeddings': True},  # set True to compute cosine similarity
    query_instruction="Represent this sentence for searching relevant passages:"
)
reranker = BCERerank(
    model="BAAI/bge-reranker-large",
    top_n=10,
    device="cuda:0",
    use_fp16=True
)
with open(f"data/musique/test.json") as file:
    questions = json.load(file)

texts = []
for q in tqdm(questions):
    texts.append(q["question"])

faiss_vectorstore = FAISS.from_texts(
    texts=texts, embedding=embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
)
faiss_retriever = faiss_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=faiss_retriever
)

retrieved_text = []
for srcdoc in retriever.get_relevant_documents(question):
    retrieved_text.append(srcdoc.page_content)

print(retrieved_text)

question3 = "What company was formed after a merger of the company that built ASM-N-5 Gorgon V?"
text3 = "\n".join([
            "The ASM-N-5 Gorgon V was an unpowered air-to-surface missile, developed by the Glenn L. Martin Company during the early 1950s for use by the United States Navy as a chemical weapon delivery vehicle.",
            "In 1995, it merged with Lockheed Corporation to form Lockheed Martin.",
            "The Yamaha DragStar 650 \"(also known as the V Star 650 and the XVS650/XVS650A)\" is a motorcycle produced by Yamaha Motor Company.",
            "The Martin Marietta Corporation was an American company founded in 1961 through the merger of Glenn L. Martin Company and American Marietta Corporation.",
            "The company was incorporated in 1917 when The Acme Tea Company merged with four small Philadelphia, Pennsylvania area grocery stores \"(Childs, George Dunlap, Bell Company,\" and \"A House That Quality Built)\" to form American Stores.",
            "The GAM-63 RASCAL was a supersonic air-to-surface missile that was developed by the Bell Aircraft Company.",
            "Ryan built several historically and technically significant aircraft, including four innovative V/STOL designs, but its most successful production aircraft was the Ryan Firebee line of unmanned drones used as target drones and unmanned air vehicles.",
            "The A-135 (NATO: ABM-3 Gorgon) anti-ballistic missile system is a Russian military complex deployed around Moscow to counter enemy missiles targeting the city or its surrounding areas.",
            "The CGR class C1 and C1a were steam locomotives of the Garratt type built by Beyer, Peacock and Company for the Ceylon Government Railways (CGR), now Sri Lanka Railways.",
            "The Alco RSD-15 was a diesel-electric locomotive of the road switcher type built by the American Locomotive Company of Schenectady, New York between August 1956 and June 1960, during which time 75 locomotives were produced."
        ])

instruction1 = "You are a helpful assistant. According to given articles, answer the given question. Think step by step. Output the executing result of every step."
prompt1 = "Question:\n" + question2 + "\n\nCorpus:\n" + text2
instruction2 = "According to the given question and the reasoning steps, output the complete final answer as short as possible. Specially, if the given question is a yes-no question, just output \"yes\" or \"no\"."
        
response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": instruction1},
        {"role": "user", "content": prompt1},
    ],
)
res1 = response.choices[0].message.content

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": instruction2},
        {"role": "user", "content": "----------Question----------\n" + question2 + "\n\n----------Steps----------\n" + res1},
    ],
)
res2 = response.choices[0].message.content

previous_answer = parse_answer(res2)

REFLECT_INSTRUCTION = "You are an advanced reasoning agent that can improve based on self reflection. You will be given a question, relevant corpus and a previous answer. Check the answer. According to the question and relevant corpus, if the previous answer is correct, output \"Correct\". Otherwise, diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure in a few sentences. Use complete sentences. \n\n------------------------------\nExample:\n\nInput:\n" + "----------Corpus----------\n" + text + "\n\n----------Question----------\n" + question + "\n\n----------Previous answer----------\nKevin Smith.\n\nOutput:\nCorrect. SModcast Pictures is an American film distribution company. SModcast Pictures was founded by Kevin Smith in 2011. So the company that distributed the film UHF is SModcast Pictures, and it was founded by Kevin Smith."

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": REFLECT_INSTRUCTION},
        {"role": "user", "content": "----------Corpus----------\n" + text2 + "\n\n----------Question----------\n" + question2 + "\n\n----------Previous answer----------\n" + previous_answer},
    ],
)
reflection = response.choices[0].message.content

if ("Correct" in reflection or "correct" in reflection) and "incorrect" not in reflection and "Incorrect" not in reflection:
    res = res2
else:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": instruction2},
            {"role": "user", "content": "----------Steps----------\n" + res1 + "\n\n" + reflection + "\n\n----------Question----------\n" + question2},
        ],
    )
    res = response.choices[0].message.content

print(res1)
print("-------------")
print(res2)
print("-------------")
print(reflection)
print("-------------")
print(res)