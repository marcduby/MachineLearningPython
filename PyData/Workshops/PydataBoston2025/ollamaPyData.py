

# imports
import ollama
import json



# methods
def test_ollama():
    '''
    Docstring for test_ollama
    '''
    pass 

def print_models():
    '''
    Docstring for print_models
    '''
    models = list(ollama.list())[0][1]

    print('\n\nOllama models (local):\n')
    for m in models:
        print(f'{m.model:<30}\t'+
            f'{m.details.family}\t'+
            f'{m.details.parameter_size}\t'+
            f'{int(m.size/(1e6)):>6} MB\t'+
            f'{m.details.quantization_level}\t'+
            f'{m.details.format}')
        

def test_model(model):
    '''
    Docstring for test_model
    
    :param prompt: Description
    '''
    response = ollama.generate(model='gemma3:270m',
                           prompt='Tell me a one paragraph story about a chicken')    
    
    return response.get('response', '')


if __name__ == "__main__":
    # print the models
    print_models()

    # test a model
    list_models = ['gemma3:270m', 'gpt-oss:20b']
    for row in list_models:
        result = test_model(model=row)
        # print("for kodel: {}, got: \n{}".format(row, json.dumps(result, indent=2)))
        print("for kodel: {}, got: \n{}".format(row, result))




