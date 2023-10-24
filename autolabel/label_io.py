import openpyxl, csv


def export_excel_to_csv(file_path, output_csv_path):
    # Open the Excel workbook
    wb = openpyxl.load_workbook(file_path)
    
    # Get all sheet names
    sheet_names = wb.sheetnames
    
    # Filter the sheet names
    filtered_sheet_names = [name for name in sheet_names if "labels" in name and "progress" not in name and "alt" not in name]
    
    # Select the last sheet name from the filtered list
    selected_sheet_name = filtered_sheet_names[-1]
    sheet = wb[selected_sheet_name]
    
    # Extract data from the selected sheet
    data = [[cell for cell in row] for row in sheet.iter_rows(values_only=True)]
    
    # Export the data to a CSV file
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)



