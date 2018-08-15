import os

def make_dic():
    # O correct predict
    with open("Mem2Seq/PUNK3.txt") as f:
        right = ""
        count_O = count_X = 0
        dic_O = {}
        dic_X = {}
        for line in f.readlines():
            #if right == "":
            #    right = line
            #    continue
            #line_token = right.split(' ')
            right = line.split(' ')[1]
            line_token = line.split(' ')
            x = ' '.join(line_token[9:]) + "$$" + ' '.join(line_token[1:9])
            #x = line.strip()+"$$"+right[1:].strip()   # predict $$ correct
            if line_token[0] == "O":
                count_O += 1
                try:
                    dic_O[line_token[1]].append(x)
                except:
                    dic_O[line_token[1]] = [x]
            elif line_token[0] == "X":
                count_X += 1
                try:
                    dic_X[line_token[1]].append(x)
                except:
                    dic_X[line_token[1]] = [x]
            right = ""
        return dic_O,dic_X,count_O, count_X

def analysis(dic1, dic2, count, entity):
    dic1 = sorted(dic1.items(), key=lambda x : len(x[1]))
    c = 0
    temp_c = 0
    for i in dic1:
        dic1_len = len(i[1])
        if i[0] in list(dic2.keys()):
            dic2_len = len(dic2[i[0]])
        else:
            dic2_len = 0
        print("{} : {}({:.2f}%)/{} {:.2f}%".format(i[0], dic1_len, dic1_len/count*100,dic2_len, dic1_len/(dic1_len + dic2_len)*100))
        slot_count = 0
        if i[0] == "what":
            for line in i[1]:
                l = line.split('$$')
                try:
                    if l[0][:-7] == l[1][:-6]:
                        c+=1
                    else:
                        print(l[0][-7] , l[1][-6])
                except:
                    pass

            print(i[1])
    print(temp_c, c)

def entity():
    dic = {}
    with open(os.path.join("Mem2Seq","data","dialog-bAbI-tasks","dialog-babi-task6-dstc2-kb.txt")) as f:
        for line in f.readlines():
            dic[line.strip().split(' ')[1]]=""
    return dic

entity_list = list(entity().keys())

dic_O , dic_X, count_O, count_X = make_dic()
analysis(dic_X, dic_O, count_X+count_O, entity_list)
print(len(dic_O), len(dic_X))
print(count_X+count_O)

