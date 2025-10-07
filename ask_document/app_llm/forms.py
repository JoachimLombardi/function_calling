from django import forms


class FunctionCallingForm(forms.Form):
    LLM_CHOICE = [
    ("gemma3", "gemma 3"), 
    ("mistral", "mistral"),
    ("mistral-small3.1", "mistral small"),
    ("llama3.2", "llama 3.2"),
    ("gpt-oss", "gpt oss"),
    ("openai/gpt-oss-20b", "gpt oss huggingface"),
    ]
    query = forms.CharField(label="Poser une question", widget=forms.TextInput(attrs={'class': 'form-control'}))
    llm_choice = forms.ChoiceField(choices=LLM_CHOICE, label="Choisissez un llm", 
                                   widget=forms.Select(attrs={'class': 'form-select'}), 
                                   required=True)
    web_search = forms.BooleanField(label="Rechercher sur internet", required=False)
    

class ImageUploadForm(FunctionCallingForm):
    image = forms.ImageField(label="Sélectionnez une image", required=True)


class PdfUploadForm(FunctionCallingForm):
    pdf = forms.FileField(label="Sélectionnez un fichier PDF", required=True)