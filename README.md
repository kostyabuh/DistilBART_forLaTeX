# DistilBART_forLaTeX
Проект является идейным продолжением проекта [EMMA](https://github.com/basic-go-ahead/emma). Репозиторий содержит использовавшиеся для реализации проекта .ipynb файлы, а также примеры использования и результатов восстановления разметки.

# Описание модели:
Модель для восстановления разметки в формате LaTeX русскоязычных текстов, содержащих математические сущности.
Модель является дообученной на переведённом&аугментированном датасете "[Mathematics Stack Exchange API Q&A Data](https://zenodo.org/records/1414384)" версией модели [sshleifer/distilbart-cnn-12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6).

# Пример использования:
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IPython.display import display, Math, Latex

model_dir = "/finetuned_model/model_bart"
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def get_latex(text):
    inputs = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        hypotheses = model.generate(
            **inputs, 
            do_sample=True, 
            top_p=0.95, 
            num_return_sequences=1, 
            repetition_penalty=1.2,
            max_length=len(text),
            temperature=0.6,
            #top_k=50,
            min_length=10,
            length_penalty=1.0,
            #num_beams=5,
            no_repeat_ngram_size=2,
            #early_stopping=True,
        )
    for h in hypotheses:
        display(Latex(tokenizer.decode(h, skip_special_tokens=True)))
        print(tokenizer.decode(h, skip_special_tokens=True))

text = 'интеграл от 3 до 5 по икс dx'
get_latex(text)
```
