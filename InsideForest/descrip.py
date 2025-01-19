import os
import re
from openai import OpenAI
import pandas as pd

def generate_descriptions(condition_list, language='en', OPENAI_API_KEY=None, default_params=None):

    client = OpenAI(api_key=OPENAI_API_KEY)

    if default_params is None:
        def get_default_params():
            return {
                'model': 'gpt-4-turbo',
                'temperature': 0.5,
                'max_tokens': 1500,
                'n': 1,
                'stop': None,
            }
        default_params = get_default_params()

    # Crear un único mensaje con todas las condiciones
    conditions_text = "\n".join([f"{i+1}. {condition}" for i, condition in enumerate(condition_list)])

    # Prompt mejorado para descripciones simples y comprensibles
    system_prompt = "You are an assistant that helps to describe dataset groups in very simple terms."
    user_prompt = (
        f"Generate a very simple description for each of the following conditions. "
        f"Use everyday language. Avoid specific numbers and ranges; instead, "
        f"use general groups like 'elderly people', 'classic cars', etc."
        f"Make each description visually friendly highlight what makes that condition unique and using emojis. Structure: 'EMOJI': 'RESPONSE'"
        f"Only respond with the descriptions in {language}. Conditions:\n\n{conditions_text}"
    )

    mensajes = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Crear una solicitud de finalización de chat con todos los mensajes
    respuesta = client.chat.completions.create(
        messages=mensajes,
        **default_params
    )

    # Dividir la respuesta en una lista de descripciones por línea
    descriptions = respuesta.choices[0].message.content.strip().split("\n")
    descriptions = [desc.strip() for desc in descriptions if desc.strip()]

    # Return a dictionary with the responses
    result = {'respuestas': descriptions}
    return result


def categorize_conditions(condition_list, df=None):
    descriptions = []

    # If df is provided, calculate thresholds using quantiles
    if df is not None:
        thresholds = {}
        for column in df.columns:
            # Calculate quantiles for low, medium, high categories
            low = df[column].quantile(0.33)
            high = df[column].quantile(0.66)
            thresholds[column] = {'low': low, 'high': high}

    for condition in condition_list:
        features = {}
        # Regex pattern to extract variable ranges
        pattern = r'(\d+\.?\d*) <= (\w+) <= (\d+\.?\d*)'
        matches = re.findall(pattern, condition)

        for match in matches:
            min_value, feature_name, max_value = match
            min_value = float(min_value)
            max_value = float(max_value)
            # Calculate average value
            avg_value = (min_value + max_value) / 2
            # Categorize based on thresholds
            if feature_name in thresholds:
                low = thresholds[feature_name]['low']
                high = thresholds[feature_name]['high']
                # Determine category based on where the average value falls within the thresholds
                if avg_value <= low:
                    category = 'BAJO'
                elif avg_value <= high:
                    category = 'MEDIO'
                else:
                    category = 'ALTO'
                features[feature_name] = category
            else:
                features[feature_name] = 'N/A'

        # Create description using the categories
        description_parts = []
        for feature, category in features.items():
            description_parts.append(f"{feature} es {category}")
        description = ', '.join(description_parts) + '.'
        descriptions.append(description)

    # Return a dictionary with the responses
    result = {'respuestas': descriptions}
    return result
