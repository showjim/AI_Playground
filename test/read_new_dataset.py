# -*- coding:utf-8 -*-

import pandas as pd

def main():
    file_path = r"C:\Users\zhouchao\OneDrive\Teradyne_Projects\02_Projects\18_TER_Chatbot\00_document\ATE Q&A - July 28.xlsx"
    list_dicts= []
    upload_qa_df = pd.read_excel(file_path, sheet_name="ATE Q&A")
    df = upload_qa_df.loc[:,["DocType", "Originator", "Status", "Title", "Expertise1", "Experstise2", "Content"]]
    df_sort=df.sort_values(by=["Title", "Status"])#,ascending=False

    df_sort_new = df_sort.copy()
    df_sort_new["Response"] = ""
    q_index = 0
    ppl_ask_q = ""
    for ind, row in df_sort.iterrows():
        if row["DocType"] == "Question":
            q_index = ind
            resp_id = 0
            ppl_ask_q = str(row["Originator"])
        elif row["DocType"] == "Response":
            df_sort_new.loc[q_index, "Response"] += str(row["Originator"]) + ": " + str(row["Content"]) + "\n"
            resp_id += 1
        else:
            print("Error: Not known DocType: " + row["DocType"] + ", in index: " + str(ind))

    df_sort_new.to_csv("result.csv")
    #list_dicts += upload_qa_df.to_dict("records")

def filter_resp():
    file_path = r"C:\Users\zhouchao\PycharmProjects\AutoMeetingMinutes\test\result_uf_select.csv"
    df = pd.read_csv(file_path)
    df_new = df[df.apply(lambda x: x["Originator"] not in x["Response"], axis=1)]
    df_new.to_csv("result_filter.csv")

def filter_top_answer_response():
    top_answ_uf = ["Mike Patnode", "Peter Turner", "Carl Peach", "Massimo Zambusi", "Keith Lucy", "Tom Vance",
                   "Allan-Demetrio Aquino", "Juanq-Long Goh", "Jacques Vieuxloup", "Edward Seng", "Jack Lee",
                   "Jong-Ik Oh", "David Ducrocq", "Edmond Shengjie Tan", "John Morris", "Ralf Baumann", "Dae-Sung Kim",
                   "Ragul Dhanavel", "Eric Varner"]
    file_path = r"C:\Users\zhouchao\PycharmProjects\AutoMeetingMinutes\test\result_filter.csv"
    list_dicts = []
    df = pd.read_csv(file_path,encoding='ISO-8859-1')
    df_new = df[df.apply(filter_func, top_answ_uf=top_answ_uf, axis=1)]
    df_new.to_csv("result_filter_slect_top_answ.csv")

def filter_func(x, top_answ_uf):
    for ppl in top_answ_uf:
        if ppl in x["Response"]:
            return True
    return False

if __name__ == "__main__":
    # main()
    # filter_resp()
    filter_top_answer_response()