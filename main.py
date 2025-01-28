import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
from Hopfield import Hopfield


#初始化全域變數
root = tk.Tk()

#tk變數宣告
EP= tk.IntVar(value=10)
PIC_ID = tk.IntVar(value = 1)
FP = tk.StringVar(value=".\\data\\Bonus_Training.txt")
FP_TEST = tk.StringVar(value=".\\data\\Bonus_Testing.txt")
RP = tk.IntVar(value = 10)
Training_status = tk.StringVar(value="尚未訓練")
Testing_status = tk.StringVar(value="\nEpoch = 0")
epoch= EP.get()

# 圖表初始化
fig1 = plt.figure()            
ax = fig1.add_subplot(111)  
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)




#選擇訓練檔案按鈕
def select():
    global matrixs_training
    Training_status.set("新檔案尚未訓練")
    Testing_status.set("\nEpoch = 0")
    FP.set(filedialog.askopenfilename()) 
    matrixs_training = read_data(FP.get())
    combobox['values'] = list(range(1,len(matrixs_training)+1))
    show_data(matrixs_training[PIC_ID.get()-1],ax,canvas)
    #print(matrixs_training)
    
    
#選擇測試檔案按鈕
def select_predict():
    global matrixs_testing
    Testing_status.set("\nEpoch = 0")
    FP_TEST.set(filedialog.askopenfilename()) 
    matrixs_testing = read_data(FP_TEST.get())
    combobox['values'] = list(range(1,len(matrixs_testing)+1))
    show_data(matrixs_training[PIC_ID.get()-1],ax,canvas)
    show_data(matrixs_testing[PIC_ID.get()-1],ax2,canvas2)


#讀取檔案
def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = []
    matrix = []
    for line in lines:
        row = []
        for char in line :  
            if char =='\n':
                continue
            row.append(1 if char == '1' else -1)
        if len(row)>0:
            matrix.append(row)
        else:
            data.append(matrix)
            matrix = []
    if len(matrix)>0:
        data.append(matrix)
    return data


#繪圖
def show_data(matrix,ax,canvas):
    matrix = np.array(matrix)
    #使用imshow繪製矩陣
    cax = ax.imshow(matrix, cmap='gray', interpolation='nearest')

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))

    #顯示網格線
    #ax.grid(which='both', color='black', linestyle='-', linewidth=2)

    # plt.show()
    canvas.draw()


#更新圖片
def load_picture(event = None):
    show_data(matrixs_training[PIC_ID.get()-1],ax,canvas)
    show_data(matrixs_testing[PIC_ID.get()-1],ax2,canvas2)
    print(matrixs_testing[PIC_ID.get()-1])


#訓練按鈕
def training():
    global Hopfield_netowrk
    Training_status.set("訓練中...")
    Hopfield_netowrk = Hopfield(matrixs_training)
    Training_status.set("訓練完成")
    

#非同步更新模式回想
def testing_async():
    show_data(matrixs_training[PIC_ID.get()-1],ax,canvas)
    if not Hopfield_netowrk:
        print("尚未進行訓練")
    else:
        Hopfield_netowrk.predict_async(matrixs_testing[PIC_ID.get()-1],max_epoch= EP.get(),ax=ax2,canvas=canvas2,status_panel=Testing_status)
    # show_data(recalled_pattern,ax,canvas)
  
  
#同步更新模式回想
def testing_sync():
    if not Hopfield_netowrk:
            print("尚未進行訓練")
    else:
        Hopfield_netowrk.predict_sync(matrixs_testing[PIC_ID.get()-1],ax=ax2,canvas=canvas2,status_panel=Testing_status)
    

#進階題:自行加入雜訊
def flip_elements(proportion):
    matrix = matrixs_testing[PIC_ID.get()-1]
    if not (0 <= proportion <= 1):
        print("錯誤輸入比例")
        return
    matrix = np.array(matrix)
    total_elements = matrix.size
    flip_nums = int(total_elements * proportion)

    indices = np.random.choice(total_elements, flip_nums, replace=False)

    flat_matrix = matrix.flatten()

    for index in indices:
        if flat_matrix[index] == 1:
            flat_matrix[index] = -1
        elif flat_matrix[index] == -1:
            flat_matrix[index] = 1

    flipped_matrix = flat_matrix.reshape(matrix.shape)
    matrixs_testing[PIC_ID.get()-1] = flipped_matrix
    show_data(matrixs_testing[PIC_ID.get()-1],ax2,canvas2)


#面板區域
Control_area = tk.Frame(root)
Control_area.grid(row = 0,column=2,sticky=tk.W+tk.E+tk.S+tk.N)
training_area = tk.LabelFrame(root,text= "正確答案圖片 (answer)")
training_area.grid(row = 0,column=0)
testing_area = tk.LabelFrame(root,text= "回想 (recall)")
testing_area.grid(row = 0,column=1)
training_result_area = tk.LabelFrame(Control_area,text= "training_status")
training_result_area.grid(row = 0,column=0,sticky=tk.W+tk.E+tk.S+tk.N)
testing_result_area = tk.LabelFrame(Control_area,text= "testing_status")
testing_result_area.grid(row = 0,column=1,sticky=tk.W+tk.E+tk.S+tk.N)
variable_area = tk.LabelFrame(Control_area,text= "Variable")
variable_area.grid(row = 1,column=0,columnspan=2,sticky=tk.W+tk.E+tk.S+tk.N)
basic_action_area = tk.LabelFrame(Control_area,text="basic actions")
advance_action_area = tk.LabelFrame(Control_area,text="advance actions")

#圖表設定
canvas = FigureCanvasTkAgg(fig1, master=training_area)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
canvas2 = FigureCanvasTkAgg(fig2, master=testing_area)  # A tk.DrawingArea.
canvas2.draw()
canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)



#變數調整區域
tk.Label(variable_area, text="max epoch").pack()
tk.Entry(variable_area, textvariable=EP).pack()
tk.Button(variable_area,text="選擇訓練資料",command=select).pack()
# tk.Label(variable_area, text="訓練資料路徑:").pack()
tk.Label(variable_area, textvariable=FP).pack()
tk.Button(variable_area,text="選擇測試資料",command=select_predict).pack()
# tk.Label(variable_area, text="測試資料路徑:").pack()
tk.Label(variable_area, textvariable=FP_TEST).pack()
tk.Label(variable_area, text="選取第幾張圖片用作測試:").pack()

pic_ids = list(range(1, 11))
combobox = ttk.Combobox(variable_area,textvariable=PIC_ID, values=pic_ids)
combobox.bind("<<ComboboxSelected>>",load_picture)
combobox.pack()



#動作action區域
basic_action_area.grid(row = 2,column=0,sticky=tk.W+tk.E+tk.S+tk.N)
advance_action_area.grid(row = 2,column=1,sticky=tk.W+tk.E+tk.S+tk.N)

tk.Button(basic_action_area,text="重設回想顯示(reset recall)",command=load_picture).pack()
tk.Label(training_result_area,textvariable=Training_status).pack()
tk.Label(testing_result_area,textvariable=Testing_status).pack()

btn = tk.Button(basic_action_area, text='訓練/train',command=training)     # 建立 Button 按鈕
btn2 = tk.Button(basic_action_area, text='測試(async)/test(async)',command=testing_async)     # 建立 Button 按鈕
btn3 = tk.Button(basic_action_area, text='測試(sync)/test(sync)',command=testing_sync)     # 建立 Button 按鈕
btn.pack()
btn2.pack()
btn3.pack()


#進階題:(自行加入雜訊並進行回想)區域:
tk.Label(advance_action_area, text="雜訊像素比例(%)").pack()
tk.Entry(advance_action_area, textvariable=RP).pack()
tk.Button(advance_action_area,text="隨機加入雜訊",command=lambda: flip_elements(RP.get()/100)).pack()


#初始化讀取圖片
matrixs_training = read_data(FP.get())
matrixs_testing = read_data(FP_TEST.get())
combobox['values'] = list(range(1,len(matrixs_testing)+1))

tk.mainloop()