import pandas as pd
import requests
import json

def load_csv(file_path):
    """
    Load CSV data from a given file path.
    """
    return pd.read_csv(file_path)

def get_json_from_url(url):
    """
    Fetch JSON data from a URL.
    """
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: Unable to retrieve data. Status code: {response.status_code}")
        return None

def parse_json_to_dataframe(json_data):
    """
    Transform JSON data into a Pandas DataFrame format.
    """
    records = []
    for intent in json_data.get('intents', []):
        tag = intent.get('tag')
        for pattern in intent.get('patterns', []):
            for response in intent.get('responses', []):
                records.append({'tag': tag, 'pattern': pattern, 'response': response})
    return pd.DataFrame(records)

def main():
    # Load CSV datasets
    patient_therapist_data = load_csv('/Users/Vimarsh/Desktop/MindSutra/data/patient_therapist_convs.csv')
    synthetic_therapy_data = load_csv('/Users/Vimarsh/Desktop/MindSutra/data/synthetic_therapy_convs.csv')

    # Fetch and prepare JSON data
    json_url = "https://storage.googleapis.com/kagglesdsdata/datasets/2594075/4429121/intents.json?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240324%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240324T233046Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=7c815915e7696c216c23ddd64f8186d299e1e7b5dc14ef7f4467dfedcbee9d6f0c3137c0c07dc2191487dad233a93ff30df52febe3bb0b6d170b9c521df79ab4075ec59adba33b967f2ab4f3d6d7660987cc250ee50ece0ba1d460b2c0793436dc0e7809b76fbae8e8cc7ef50aa2447456451bdfd655639f2c0fbf741975d44570cef66030ab29090b431263662c2e6d752da4279d4da3e5bfe11a3b3a1f8ff209bd01dbcb22775e2b2be420330ee691b5f5debd58179432c8da6fab55b7e16b528566edcf2259726f3a0a7f670bf4be42fcc4a120623af129e46b3e415e467a5f2e8a4fed7d269d9ba4535cc76c11b8526c4939ae4a32d2b0771600177a1551"
    json_content = get_json_from_url(json_url)
    if json_content:
        intents_dataframe = parse_json_to_dataframe(json_content)

        # Print dataframes to validate loaded data
        print(patient_therapist_data.head())
        print(synthetic_therapy_data.head())
        print(intents_dataframe.head())

if __name__ == "__main__":
    main()
