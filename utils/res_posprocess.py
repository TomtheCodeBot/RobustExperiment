import os
import json

def readjust_result(result_path,results_notattacked):
    results = os.listdir(result_path)
    for result in results_notattacked.keys() :
        f = open(f'{result_path}/{result}.json')
        curren_result = json.load(f)
        already_targetted = curren_result["Total Attacked Instances"] - results_notattacked[result]
        curren_result["Successful Instances"] =int( curren_result["Successful Instances"]-already_targetted)
        curren_result["Total Attacked Instances"] = int(results_notattacked[result])
        curren_result["Attack Success Rate"] = float(curren_result["Successful Instances"]/curren_result["Total Attacked Instances"])
        f.close()
        print(curren_result)
        with open(f'{result_path}/{result}_readjusted.json', 'w') as f:
            json.dump(curren_result, f)
    pass