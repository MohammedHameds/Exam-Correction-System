from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from langdetect import detect

def initialize_model():
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-large')
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-large')
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return t5_model, t5_tokenizer, sbert_model

def compare_answers(teacher_answer, student_answer):
    t5_model, t5_tokenizer, sbert_model = initialize_model()
    task = f"paraphrase: {teacher_answer} and {student_answer} mean the same?"
    inputs = t5_tokenizer.encode(task, return_tensors='pt', max_length=512, truncation=True)
    t5_outputs = t5_model.generate(inputs)
    t5_decoded_output = t5_tokenizer.decode(t5_outputs[0], skip_special_tokens=True)

    teacher_embedding = sbert_model.encode(teacher_answer, convert_to_tensor=True)
    student_embedding = sbert_model.encode(student_answer, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(teacher_embedding, student_embedding)
    similarity_percentage = similarity_score.item() * 100

    if "yes" in t5_decoded_output.lower() or "same" in t5_decoded_output.lower():
        evaluation_result = "The meanings of both answers are aligned"
    else:
        evaluation_result = "The meanings of the answers differ"

    return evaluation_result, similarity_percentage
