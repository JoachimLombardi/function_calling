from pathlib import Path
from django.shortcuts import render
from ask_document.config import MEDIA_ROOT, MEDIA_URL
from .utils import save_file
from .business_logic import function_calling, image_questioning_llm, pdf_questioning_llm
from .forms import FunctionCallingForm, ImageUploadForm, PdfUploadForm
from django.contrib import messages
import markdown



def image_questioning(request):
    form = ImageUploadForm(request.POST or None, request.FILES)
    if request.method == 'POST':
        if form.is_valid():
            image_file = form.cleaned_data['image']
            save_path = 'data/jpg/image.jpg'
            save_file(save_path, image_file)
            query = form.cleaned_data['query']
            llm_choice = form.cleaned_data['llm_choice']
            image_url = MEDIA_URL + save_path
            try:
                response = image_questioning_llm(llm_choice, query)
                response = markdown.markdown(response)
            except Exception as e:
                messages.error(request, str(e))
                response = ""
            return render(request, 'image_questioning.html', {'form': form, 'response': response, 'image_url': image_url})
        else:
            messages.error(request, "Le formulaire n'est pas valide.")
    return render(request, 'image_questioning.html', {'form': form})


def llm_choice(request):
    form = FunctionCallingForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            query = form.cleaned_data.get('query')
            llm_choice = form.cleaned_data.get('llm_choice')
            response, context, mails = function_calling(query, llm_choice)
            if "error" in response:
                messages.error(request, response["error"])
                response = ""
            response = markdown.markdown(response, extensions=['markdown.extensions.fenced_code'])
            context = markdown.markdown(context, extensions=['markdown.extensions.fenced_code'])
            return render(request, 'function_calling.html', {'form': form, 'response': response, 'context': context, 'mails': mails})
        else:
            messages.error(request, "Le formulaire est invalide.")
    return render(request, 'function_calling.html', {'form': form})


def pdf_questioning(request):
    form = PdfUploadForm(request.POST or None, request.FILES)
    if request.method == 'POST':
        if form.is_valid():
            save_path = 'pdf/text.pdf'
            query = form.cleaned_data['query']
            llm_choice = form.cleaned_data['llm_choice']
            pdf_file = form.cleaned_data['pdf']
            pdf_path = Path(MEDIA_ROOT).joinpath(save_path)
            # Écriture du fichier
            with open(pdf_path, "wb") as f:
                for chunk in form.cleaned_data["pdf"].chunks():
                    f.write(chunk)
            pdf_url = request.build_absolute_uri(MEDIA_URL + save_path)
            print("DEBUG pdf_url:", pdf_url)
            try:
                response, context = pdf_questioning_llm(llm_choice, query, pdf_path)
                response = markdown.markdown(response)
                context = markdown.markdown(context)
            except Exception as e:
                messages.error(request, str(e))
                response = ""
                context = ""
                pdf_url = ""
            return render(request, 'pdf_questioning.html', {'form': form, 'response': response, 'context': context, 'pdf_url': pdf_url})
        else:
            messages.error(request, "Le formulaire n'est pas valide.")
    return render(request, 'pdf_questioning.html', {'form': form})