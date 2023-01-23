import xlsxwriter
from glob import glob
import json
# Create an new Excel file and add a worksheet.
def draw_excel(path_to_results):
    workbook = xlsxwriter.Workbook(path_to_results.split("/")[-1]+".xlsx")
    worksheet = workbook.add_worksheet()
    cellformat = workbook.add_format()
    cellformat.set_align('center')
    cellformat.set_align('vcenter')
    
    lst_attack = glob(path_to_results+"/*/", recursive = True)
    f = open(path_to_results+'/result.json')
    clean_acc = json.load(f)
    f.close()
    first_collumn = 65
    worksheet.merge_range(f'{chr(first_collumn)}1:{chr(first_collumn)}2',"Models",cell_format=cellformat)
    worksheet.merge_range(f'{chr(first_collumn+1)}1:{chr(first_collumn+1)}2',"Clean Accuracy (%)",cell_format=cellformat)
    for i in range(0,len(lst_attack)):
        # Merge 3 cells.
        worksheet.merge_range(f'{chr(first_collumn+(i*2)+2)}1:{chr(first_collumn+(i*2)+3)}1',lst_attack[i].split("/")[-2],cell_format=cellformat)
        worksheet.write(f'{chr(first_collumn+(i*2)+2)}2', 'ASR(%)↓',cellformat)
        worksheet.write(f'{chr(first_collumn+(i*2)+3)}2', 'Avg. Query↓',cellformat)
        results = glob(lst_attack[i]+"*.json", recursive = True)
        results = sorted(results,reverse=False)
        print(results)
        for k in range(len(results)):
            # Opening JSON file
            f = open(results[k])
            name = results[k].split("/")[-1].split(".")[0]
            data = json.load(f)
            f.close()
            worksheet.write(f'{chr(first_collumn)}{str(k+3)}', name,cellformat)
            percentage  = data["Attack Success Rate"]*100
            format_float = "{:.2f}".format(percentage)
            worksheet.write(f'{chr(first_collumn+(i*2)+2)}{str(k+3)}', f"{format_float}%",cellformat)
            querries  = data["Avg. Victim Model Queries"]
            format_float = "{:.2f}".format(querries)
            worksheet.write(f'{chr(first_collumn+(i*2)+3)}{str(k+3)}', f"{format_float}",cellformat)
            worksheet.write(f'{chr(first_collumn+1)}{str(k+3)}', clean_acc[name],cellformat)

    worksheet.autofit()
    workbook.close()
    pass

if __name__ == "__main__":
    draw_excel("/home/ubuntu/Robustness_Gym/result_official/AGNEWS")
    draw_excel("/home/ubuntu/Robustness_Gym/result_official/IMDB")
    draw_excel("/home/ubuntu/Robustness_Gym/result_official/MR")
    draw_excel("/home/ubuntu/Robustness_Gym/result_official/SST2")
    draw_excel("/home/ubuntu/Robustness_Gym/result_official/YELP")