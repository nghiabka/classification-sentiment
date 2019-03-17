import csv
import  os


def get_data(path_read_folder,path_write_folder):
    list_file = os.listdir(path_read_folder)
    # ghi vào file csv
    csvfile = open("./data_final/data_origin.csv","w")
    fieldnames = ['label', 'data']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for file in list_file:
        path_file = os.path.join(path_read_folder,file)

        if os.path.isfile(path_file):
            path_write_file = os.path.join(path_write_folder,file)
            if file =="pos.txt":
                label = 1
            elif file == "neg.txt":
                label =2
            else:
                label =3

            fr = open(path_file,"r")
            fw = open(path_write_file,"w+")
            for line in fr:
                if line[0] =="/" or len(line)<2:
                    continue
                fw.write(line.strip("\n"))
                writer.writerow({'label': label, 'data': line.strip("\n")})

            fr.close()
            fw.close()




get_data("./data_review2","./data_final")