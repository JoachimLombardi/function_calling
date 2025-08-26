import base64
import functools
import json
import os
from pathlib import Path
import re
import unicodedata
from ollama import chat
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from bs4 import BeautifulSoup
from openai import OpenAI
import requests
import fitz
from PIL import Image
from ask_document.config import MEDIA_ROOT
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
import joblib
import numpy as np
from django.conf import settings


def image_questioning_llm(llm_choice, query, path="data/jpg/image.jpg"):
    """
    A function that uses the Ollama Chat API to ask a given query about an image.
    
    Args:
        llm_choice (str): The model to use for the chat.
        query (str): The question to ask about the image.
        path (str, optional): The path to the image file. Defaults to "data/jpg/image.jpg".
    
    Returns:
        str: The response from the chat API.
    """
    response = chat(
    model=llm_choice,
    messages=[
        {
        'role': 'user',
        'content': query,
        'images': [path],
        }
    ],
    )
    return response["message"]["content"]


def function_calling(query, model):
    """
    This function is used to interact with the Ollama Chat API to ask questions about emails and car prices.
    It takes in two parameters:
    - query (str): The question to ask about the emails or car prices.
    - model (str): The model to use for the chat.
    
    This function returns a tuple containing the response from the chat API, the context of the chat, and the list of emails.
    """
    def get_message_body(payload):
        """
        Retrieves the message body from a given payload.
        
        Parameters:
            payload (dict): The payload containing the message body.
            
        Returns:
            str: The message body as plain text. If no readable content is found, returns "(pas de contenu lisible)".
        """
        if 'parts' in payload:
            for part in payload['parts']:
                mime_type = part.get('mimeType')
                body_data = part.get('body', {}).get('data')
                if body_data:
                    try:
                        decoded_data = base64.urlsafe_b64decode(body_data).decode('utf-8')
                        if mime_type == 'text/html':
                            # Extraire le texte brut du HTML
                            soup = BeautifulSoup(decoded_data, 'html.parser')
                            return soup.get_text(separator='\n').strip()
                        else:
                            return decoded_data
                    except Exception:
                        continue
        else:
            body_data = payload.get('body', {}).get('data')
            if body_data:
                decoded_data = base64.urlsafe_b64decode(body_data).decode('utf-8')
                # Ici aussi, possibilité de vérifier le type si tu veux
                soup = BeautifulSoup(decoded_data, 'html.parser')
                return soup.get_text(separator='\n').strip()
        return "(pas de contenu lisible)"


    def google_mail(number_of_mails=5):
        """
        This function retrieves the most recent personal emails from the Gmail API and returns a list of dictionaries containing the sender, subject, and content of each email.
        
        Parameters:
            number_of_mails (int, optional): The number of most recent emails to retrieve. Defaults to 5.
            
        Returns:
            list: A list containing two elements. The first element is a string containing the sender, subject, and content of each email, separated by newlines. The second element is a list of dictionaries, where each dictionary contains the sender, subject, and content of an email.
        """
        SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())

        service = build('gmail', 'v1', credentials=creds)
        try:
            results = service.users().messages().list(userId='me', maxResults=number_of_mails, labelIds=['CATEGORY_PERSONAL']).execute()
            messages = results.get('messages', [])
            liste_messages = []
            liste_mails = []
            liste_general = []
            for i, msg in enumerate(messages):
                msg_id = msg['id']
                message = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
                headers = message.get('payload', {}).get('headers', [])
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '(sans sujet)')
                sender = next((h['value'] for h in headers if h['name'] == 'From'), '(expéditeur inconnu)')
                content = get_message_body(message.get('payload', {}))
                liste_messages.append(f"Mail {i+1}:\nExpéditeur: {sender}\nSujet: {subject}\nContenu: {content}")
                message_dict = {
                    'sender': sender,
                    'subject': subject,
                    'content': content
                }
                liste_mails.append(message_dict)
            tot_mails = "\n".join(liste_messages)
            liste_general.append(tot_mails)
            liste_general.append(liste_mails)
            return liste_general
        except Exception as e:
            print(f"Une erreur s'est produite : {str(e)}")
            return []

    data = pd.read_csv("data/csv/car-sales-extended-missing-data.csv")

    def create_best_model(data):  
        """
        Create the best model based on the given data.

        Parameters:
            data (pandas.DataFrame): The data to create the model from.

        Returns:
            str: The name of the best model with the best parameters on both the train and test set,
            and the name of the best model with the best parameters on the validation set.
        """
        np.random.seed(42)

        try:
            data.dropna(subset=["Price"], inplace=True)
        except:
            raise ValueError("La colonne Price est absente des données.")
             
        # Split data
        X = data.drop("Price", axis=1)
        y = data["Price"]
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

        # Define differents features and transformer pipeline
        categorical_features = ["Make", "Colour"]
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")), # Imputer fills missing data.
            ("onehot", OneHotEncoder(handle_unknown="ignore"))]) # OneHotEncoder convert data to numbers.
        door_features = ["Doors"]
        door_transformer =  Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=4))])
        numerical_features = ["Odometer (KM)"]
        numerical_transformers = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean"))])

        # Setup preprocessing steps (fill missing values, then convert to numbers)
        preprocessor = ColumnTransformer(
                            transformers=[
                                ("cat", categorical_transformer, categorical_features),
                                ("door", door_transformer, door_features),
                                ("num", numerical_transformers, numerical_features)
                            ])
        # Setup different models
        ridge_grid = {
        "solver": ["auto"],
        "max_iter": [1000],
        "alpha": [0.1],
        }

        lasso_grid = {
        "max_iter": [1000],
        }


        rf_reg_grid = {
        "n_estimators": [100, 400],
        "min_samples_split": [2],
        "min_samples_leaf": [1, 5],
        }


        SGD_reg_grid = {
        "loss": ["squared_error"],
        "max_iter": list(range(100, 200)),
        "penalty": ["elasticnet"],
        "tol": [1e-3]
        }
        
        DTR_grid = {
        "splitter": ["best", "random"],
        "max_depth": list(range(2, 10, 5)),
        "max_leaf_nodes": [10]
        }

        models_reg = {
        "Random Forest Regressor": RandomForestRegressor(random_state=1, n_jobs=-1),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "SGD Regressor": SGDRegressor(random_state=1),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=1)
        }

        grids = {
        "Random Forest Regressor": rf_reg_grid,
        "Ridge Regression": ridge_grid,
        "Lasso Regression": lasso_grid,
        "SGD Regressor": SGD_reg_grid,
        "Decision Tree Regressor": DTR_grid
        }

        scoring = {
        'r2': 'r2',
        'neg_mean_squared_error': 'neg_mean_squared_error',
        'neg_mean_absolute_error': 'neg_mean_absolute_error',
        'neg_root_mean_squared_error': 'neg_root_mean_squared_error'
        }



        model_scores = {}
        model_best_params = {}
        model_final_scores = {}
        for name, model_reg in models_reg.items():
            try:
                # Create a preprocessing and modeling pipeline
                model = Pipeline(steps=[("preprocessor", preprocessor),
                                        ("model", model_reg)])
                param_grid = {"model__" + key: val for key, val in grids[name].items()}
                rs = GridSearchCV(model,
                                param_grid=param_grid,
                                scoring=scoring,
                                refit="r2",
                                cv=5,
                                verbose=True)
                try:
                    rs.fit(X_train, y_train)
                except Exception as e:
                    print(f"Erreur lors de l'entrainement du modèle {name}: {e}")
                model_best_params[name] = rs.best_params_
                model_scores[name] = {
                "r2": rs.cv_results_["mean_test_r2"][rs.best_index_],
                "neg_mean_squared_error": rs.cv_results_["mean_test_neg_mean_squared_error"][rs.best_index_],
                "neg_mean_absolute_error": rs.cv_results_["mean_test_neg_mean_absolute_error"][rs.best_index_],
                "neg_root_mean_squared_error": rs.cv_results_["mean_test_neg_root_mean_squared_error"][rs.best_index_]
                }
                #rs.cv_results_
                model_with_best_params = rs.best_estimator_  # modèle entraîné avec meilleurs paramètres

                # Prédictions sur le jeu de test
                y_pred = model_with_best_params.predict(X_valid)

                # Calcul des métriques
                r2 = r2_score(y_valid, y_pred)
                mse = mean_squared_error(y_valid, y_pred)
                rmse = root_mean_squared_error(y_valid, y_pred)
                mae = mean_absolute_error(y_valid, y_pred)

                model_final_scores[name] = {
                    "r2": r2,
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae
                }
                if model_with_best_params:
                    # Sauvegarder le modèle avec les meilleurs paramètres
                    model_path = Path(MEDIA_ROOT + f'/models/{name}.joblib')
                    joblib.dump(model_with_best_params, model_path)
            except Exception as e:
                print(f"Erreur avec le modèle {name}: {e}")
                model_scores = None
                model_best_params = None
                model_final_scores = None
        # Sauvegarder les scores et les meilleurs paramètres
        if model_scores:
            model_scores = dict(sorted(model_scores.items(), key=lambda item: item[1]['r2'], reverse=True))
            best_model = max(model_scores.items(), key=lambda x: x[1]["r2"])
            best_model_name = best_model[0]
            model_with_best_params_scores = Path(MEDIA_ROOT + '/json/model_with_best_params_scores.json')
            with open(model_with_best_params_scores, 'w') as f:
                json.dump(model_scores, f)
        if model_best_params:
            best_model_params = Path(MEDIA_ROOT + f'/json/best_model_params.json')
            with open(best_model_params, 'w') as f:
                json.dump(model_best_params, f)
        if model_final_scores:
            model_final_scores = dict(sorted(model_final_scores.items(), key=lambda item: item[1]['r2'], reverse=True))
            best_model_final = max(model_final_scores.items(), key=lambda x: x[1]["r2"])
            best_model_name_final = best_model_final[0]
            model_with_best_params_final_scores = Path(MEDIA_ROOT + '/json/model_with_best_params_final_scores.json')
            with open(model_with_best_params_final_scores, 'w') as f:
                json.dump(model_final_scores, f)
        return f"Best model with best params on train and test set: {best_model_name}, Best model with best params on validation set: {best_model_name_final}"


    def make_predictions(make, color, odometer, doors): 
        """
        Makes a prediction on the price of a car based on its make, color, odometer reading, and number of doors.
        
        Args:
            make (str): The make of the car.
            color (str): The color of the car.
            odometer (float): The odometer reading of the car.
            doors (int): The number of doors the car has.
            
        Returns:
            str: The predicted price of the car in dollars.
        """
        # Load the best model
        with open(MEDIA_ROOT + '/json/model_with_best_params_scores.json') as f:
            model_scores = json.load(f)
        # Select the best model
        best_model_name = max(model_scores.items(), key=lambda x: x[1]["r2"])[0]
        model_path = Path(MEDIA_ROOT + f'/models/{best_model_name}.joblib')
        model = joblib.load(model_path)
        # Mapping colors
        mapping = {
        "roug": "red",
        "vert": "green",
        "bleu": "blue",
        "noir": "black",
        "blanc": "white",
        }
        for key in mapping:
            if color.lower().startswith(key):
                color = mapping.get(key)
                break
        # Make predictions
        X_test = pd.DataFrame({"Make": [make], "Colour": [color], "Odometer (KM)": [odometer], "Doors": [float(doors)]})
        y_pred = str(int(round(model.predict(X_test)[0])))
        return f"The predicted price is {y_pred} dollars"
    # List of tools
    tools = [
        {
            "type": "function",
            "function":{
            "name": "google_mail",
            "description": "Gets mails from your Gmail account",
            "parameters": {
                "type": "object",
                "properties": {
                    "number_of_mails": {
                        "type": "integer",
                        "description": "The number of mails to get",
                        }
                    },
                "required": []
                }
            }
        },

        {
            "type": "function",
            "function":{
            "name": "create_best_model",
            "description": "Train and test models with various parameters to determine the best model",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
                }
            }
        },

        {
            "type": "function",
            "function":{
            "name": "make_predictions",
            "description": "Predict the price of a car based on its make, color, odometer and number of doors",
            "parameters": {
                "type": "object",
                "properties": {
                    "make": {
                        "type": "string",
                        "description": "The make of the car",
                        },
                    "color": {
                        "type": "string",
                        "description": "The color of the car",
                        },
                    "odometer": {
                        "type": "integer",
                        "description": "The odometer of the car",
                        },
                    "doors": {
                        "type": "integer",
                        "description": "The number of doors of the car",
                        }
                    },
                "required": ["make", "color", "odometer", "doors"]
                }
            }
        }
    ]
    # Dictionary of functions
    names_to_functions = {
        "google_mail": functools.partial(google_mail),
        "create_best_model": functools.partial(create_best_model, data),
        "make_predictions": functools.partial(make_predictions)
    }
    
    def build_messages(query):
            """
            Build the messages to be used in the conversation.

            Args:
                query (str): The query of the user.

            Returns:
                list: A list of messages to be used in the conversation.
            """
            return [
            {"role": "system", "content": "Tu es un assistant expert. Utilise uniquement la sortie des fonctions tools pour répondre."},
            {"role": "user", "content": query}
            ]


    def extract_tool_call(tool_call):
        """
        Extract the function name and its arguments from a tool call (either a ToolCall object or a dictionary).

        Args:
            tool_call (ToolCall or dict): The tool call.

        Returns:
            tuple: A tuple containing the function name and its arguments.
        """
        function_name = getattr(tool_call["function"], "name", None) if isinstance(tool_call, dict) else tool_call.function.name
        args = getattr(tool_call["function"], "arguments", None) if isinstance(tool_call, dict) else tool_call.function.arguments
        try:
            params = json.loads(args)
        except:
            params = args
        return function_name, params


    def execute_tool(function_name, params):
        """
        Execute a tool function given its name and arguments.

        Args:
            function_name (str): The name of the tool function.
            params (dict): The arguments of the tool function.

        Returns:
            tuple: A tuple containing the result of the tool function, the context and the mails.
        """
        result = names_to_functions[function_name](**params)
        mails = []
        context = result
        if function_name == "google_mail":
            mails = result[1]
            result = result[0]
        if isinstance(result, dict):
            result = json.dumps(result)
        return result, context, mails
    
    messages = build_messages(query)

    data = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "stream": False,
                "temperature": 0,
                "max_tokens": 5000,
                "top_p": 1e-6,
                "seed": 42
            }

    # Function calling
    for attempt in range(1,4):
        # --- Sélection back-end (Ollama vs HF) ---
        try:
            if settings.IS_RENDER:
                print("api huggingface call")
                client = OpenAI(base_url="https://router.huggingface.co/v1",
                                api_key=settings.HUGGINGFACE_API_KEY)
                call_fn = lambda d: client.chat.completions.create(**d)
                get_message = lambda resp: resp.choices[0].message
                add_tool_msg = lambda tool_call_id, fn, res: {
                    "tool_call_id": tool_call_id, "role": "tool", "name": fn, "content": res
                }
            else:
                print("api ollama call")
                call_fn = lambda d: requests.post('http://localhost:11434/api/chat', json=d).json()
                get_message = lambda resp: resp["message"]
                add_tool_msg = lambda _id, fn, res: {
                    "role": "tool", "name": fn, "content": res
                }

            # Premier appel
            response = call_fn(data)
            first_message = get_message(response)
            messages.append(first_message)

            # Extraction tool
            tool_call = first_message.tool_calls[0] if settings.IS_RENDER else first_message["tool_calls"][0]
            function_name, params = extract_tool_call(tool_call)
            print("function_name:", function_name, "params:", params)

            # Exécution tool
            tool_result, context, mails = execute_tool(function_name, params)
            tool_msg = add_tool_msg(getattr(tool_call, "id", None), function_name, tool_result)
            messages.append(tool_msg)

            # Deuxième appel
            data = {"model": model, "messages": messages, "stream": False}
            response = call_fn(data)
            final_content = (get_message(response).content if settings.IS_RENDER else response["message"]["content"])
            return final_content, context, mails
        except Exception as e:
            print(f"LLM call failed with error: {e}")
    return response, context, mails
     

def reorder_text_pdf(_context_, file_write, list_path, model, query):
    """
    Given a list of text paths and a query, this function uses the Mistral API to proofread and correct the text for spelling, grammar, and formatting errors while preserving its original meaning. It ensures proper spacing between words, corrects any misplaced line breaks, and maintains readability.

    Parameters:
    - _context_ (list): A list of text paths
    - file_write (str): The path to write the corrected text to
    - list_path (list): A list of text paths
    - model (str): The model to use for the Mistral API call
    - query (str): The query to answer

    Returns:
    - dict: A dictionary containing the answer to the query
    """
    list_url = []
    for path in list_path:
        print(path)
        try:
            with open(path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            print(f"Error: The file {path} was not found.")
            return None
        except Exception as e:  
            print(f"Error: {e}")
            return None
        list_url.append(base64_image)
    template = """Please proofread and correct the following French text for spelling, grammar, and formatting errors while preserving its original meaning. 
    Ensure proper spacing between words, correct any misplaced line breaks, and maintain readability. Do not alter the style or tone of the original text. 
    Ignore headers, footers, and any repetitive elements appearing on each page.
    If necessary, make minor adjustments for clarity.

    The following text corresponds to images. For each portion of the text, make sure to consider the image as a reference for context.

    ## Output_format:
    {
        "answer":string
    }
    """
    for attempt in range(1,4):
        try:
            messages = [{"role":"user", "content": template, "images": []}]
            data =  {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "echo": False,
                    "max_tokens": 5000,
                    "temperature": 0,
                    "top_p": 1e-6,
                    "top_k": -1,
                    "logprobs": None,
                    "n": 1,
                    "best_of": 1,
                    "use_beam_search": False,
                    "seed":42,
            }
            for part_text, image_url in zip(_context_, list_url):     
                messages[0]["content"] += "\n\n" + part_text
                messages[0]["images"].extend([image_url])
            response = requests.post('http://localhost:11434/api/chat', json=data).json()
            print("===============\n", response)
            context = response["message"]["content"]
            context = json.loads(context)
            context = context.get("answer").replace("\n", " ").replace("\u202f", " ").strip()
            print("=================================================================\n", context)
            template = """ You are a document reader, you will be given a text and a query. Your task is to answer the query based on the text.

            ## Instructions:
            1. Read the text and the query.
            2. Answer the query based on the text.
            3. Give the answer in a json format.
            
            ## Text:
            '"""+context+"""'

            ## Query:
            '"""+query+"""'

            ## Output_format:
            {
                "answer":string
            }
            """
            messages = [{"role":"user", "content": template}]
            data =  {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "echo": False,
                    "max_tokens": 5000,
                    "temperature": 0,
                    "top_p": 1e-6,
                    "top_k": -1,
                    "logprobs": None,
                    "n": 1,
                    "best_of": 1,
                    "use_beam_search": False,
                    "seed":42,
            }
            response = requests.post('http://localhost:11434/api/chat', json=data).json()
            print("===============\n", response)
            response = response["message"]["content"]
            print("=================================================================\n response", response)
            try:
                response = json.loads(response)
                response = response.get("answer")
                print("=================================================================\n response", response)
            except json.decoder.JSONDecodeError:
                pass
            return response, context
        except Exception as e:
            print(f"Mistral API call failed with error: {e}")
            print(f"Attempt {attempt} failed. Retrying...")


def pdf_questioning_llm(model, query, path):
    """
    Convert a PDF document to a list of images and perform a text questioning task on it using the Mistral API.
    
    Args:
        model (str): The model to use for the text questioning task.
        query (str): The query to ask about the PDF document.
        path (str): The path to the PDF document.
        
    Returns:
        dict or Exception: The answer to the query in a JSON format, or an Exception if the API call fails.
    """
    output_file = f"data/json/text.json"
    doc = fitz.open(path)
    list_img_path = []
    text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # Extract text
        text.append(page.get_text())
        # Image path
        img_path = path.parent / f"{path.stem}_{page_num}.jpg"
        list_img_path.append(img_path)
        # Convert to image
        matrix = fitz.Matrix(2, 2) 
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # Save image
        img.save(img_path, "JPEG", quality=100)
    doc.close()
    try:
        return reorder_text_pdf(text, output_file, list_img_path, model, query)
    except Exception as e:
        return e




