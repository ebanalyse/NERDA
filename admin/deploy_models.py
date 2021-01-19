from NERDA.datasets import get_conll_data, get_dane_data
import pandas as pd
import torch
import boto3

def deploy_model_to_s3(model, test_set = get_dane_data('test')):
    """Deploy Model to S3

    Args:
        model: NERDA model.
        test_set: Test set for evaluating performance.

    Returns:
        str: message saying, if model was uploaded successfully. 
        Model and text file with performance numbers uploaded 
        as side-effects.
    """

    model_name = type(model).__name__

    file_model = f'{model_name}.bin'
    torch.save(model.network.state_dict(), file_model)
    
    # compute performance on test set and save.
    performance = model.evaluate_performance(test_set)
    
    # write to csv.
    file_performance = f'{model_name}_performance.csv'
    performance.to_csv(file_performance, index = False)
    
    # upload to S3 bucket.
    s3 = boto3.resource('s3')
    s3.Bucket('nerda').upload_file(
            Filename=file_model, 
            Key = file_model)
    s3.Bucket('nerda').upload_file(
            Filename=file_performance, 
            Key = file_performance)

    return "Model deployed to S3 successfully." 

if __name__ == '__main__':
    from NERDA.precooked import EN_ELECTRA_EN
    model = EN_ELECTRA_EN()
    model.train()  

    deploy_model_to_s3(model)

