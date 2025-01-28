# 實作離散Hopfield網路
113522053 蔡尚融

實作加分題:
1. Bonus_Training Set
1. 自行加入雜訊進行回想




# 程式執行操作說明


程式介面:
![image](https://hackmd.io/_uploads/H1DDtgEN1l.png)


1. 擇訓練、測試資料檔案:
![image](https://hackmd.io/_uploads/B1bhbgENJg.png)
選擇完檔案後介面:
![image](https://hackmd.io/_uploads/HyUL7eVNkg.png)
2. 點選訓練按鈕開始訓練Hopfield網路
![image](https://hackmd.io/_uploads/SyWk4e4N1x.png)

3. 選擇使用測試及中的哪張圖片測試
![image](https://hackmd.io/_uploads/rJEMOeE4ke.png)



4. 開始進行回想
![image](https://hackmd.io/_uploads/ByFAVgNNJx.png)  

>使用者可以選擇使用同步/非同步更新的方式來進行網路回想recall。  
>* 同步回想模式中整張圖片(輸入matrix)會一起更新。  
>* 非同步回想模式中matrix中的每個像素會逐一更新。  

5. 回想完成
![image](https://hackmd.io/_uploads/HJ5r8eVN1g.png)



# 程式簡介
## Hopfield.py(網路部分)
實作Hopfield網路
:::spoiler 程式碼
```python= 1
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
import time

class Hopfield:
    w=[]
    theta = []
    def __init__(self,matrixs):
        print("start training")
        matrixs = np.array(matrixs)
        n = len(matrixs)
        self.j_row = len(matrixs[0])
        self.j_column = len(matrixs[0][0])
        self.p = self.j_row*self.j_column
        print(f"array shape = {self.j_row},{self.j_column}")

        print(f"len = {self.p}")
        self.w = np.zeros((self.p, self.p))
        for matrix in matrixs:
            print(np.array(matrix))
            matrix =np.array([matrix.flatten()]).T
            #print("flatten = \n",matrix)
            #print(f"transpose : \n{matrix.T}")
            self.w += np.dot(matrix ,matrix.T)
        
        self.w = self.w/self.p - n/self.p*np.eye(self.p)
        print(self.w)
        self.theta = np.array([np.sum(self.w, axis=1)]).T
        #print("theta:",self.theta)
        print("traning complete")
        
        
    def predict_sync(self,matrix,ax = None,canvas = None,status_panel = None):
        print("start testing(sync).")
        last_result = [-1]
        result = []
        matrix = self.array_flatten(matrix)
        while not np.array_equal(last_result,result):
            self.recall_status_panel_update(status_panel)  
            last_result = result.copy()
            result = []
            
            result = np.dot(self.w,matrix)
            # print("result dot:\n",result)
            #result = result - self.theta
            # print("result -theta:\n",result)
            
    
            for j,row in enumerate(result):
                result[j][0] =self.sign(row[0],matrix[j][0])
        
            #print("result sign:\n",result)
            self.refresh_picture(result.reshape(self.j_row,self.j_column),ax,canvas)
            matrix = result.copy()
        
        self.recall_status_panel_update(status_panel,done=True)               
        print("testing complete(sync).")
        return result.reshape(self.j_row,self.j_column)
    
    def predict_async(self,matrix,max_epoch = 10,ax = None,canvas = None,status_panel = None):
        print("start testing(async).")
        matrix = self.array_flatten(matrix)
        last_result = []
        #print(matrix)
        for epoch in range(max_epoch):
            self.recall_status_panel_update(status_panel,epoch)
            if np.array_equal(matrix,last_result):
                max_epoch = epoch
                break
            last_result = matrix.copy()
            print(f"epoch : {epoch}")
            for j in range(self.p):
                y = 0
                # for i in range(self.p):
                #     y += self.w[j][i] * matrix[i][0]
                y = np.dot(self.w[j],matrix)
                #y -= self.theta[j][0]
                matrix[j][0] = self.sign(y,matrix[j][0])
                self.refresh_picture(matrix.reshape(self.j_row,self.j_column),ax,canvas)
            
        print("testing complete(async).")
        self.recall_status_panel_update(status_panel,max_epoch,True)
        self.refresh_picture(matrix.reshape(self.j_row,self.j_column),ax,canvas)

        return matrix.reshape(self.j_row,self.j_column)
    
    
    
    def recall_status_panel_update(self,status_panel,epoch = None,done = False):
        if status_panel:
            if not done:
                status_panel.set(f"\n回想中...\n")
            if done:
                status_panel.set(f"\n回想完成\n")
                
            if epoch!=None:
                status_panel.set(status_panel.get()+f"Epoch = {epoch}\n")
    
    
    def sign(self,value,last_value):
        if value > 0:
            return 1
        elif value == 0:
            return last_value
        else:
            return -1
        
    def array_flatten(self,matrix):
        matrix = np.array(matrix)
        matrix = np.array([matrix.flatten()])
        matrix = matrix.T
        return matrix
    
    def refresh_picture(self,result,ax,canvas):
        if (ax!=None) and (canvas!=None):
            #print("refresh!")
            ax.cla()
            ax.imshow(result, cmap='gray', interpolation='nearest')
            canvas.draw()
            canvas.flush_events()
            #time.sleep(0.05)
            
    

        
    
        
        
#測試輸入    
if __name__ == "__main__":
    
    matrix = [[[1],[-1],[1]],[[-1],[1],[-1]]]
    
    test = Hopfield(matrix)
    result = test.predict_async([[1],[1],[-1]])
    print(f"final result :\n {result}")

```
:::
### class Hopfield：
每當使用者點擊"訓練/training"按鈕時，就會建立一個新的Hopfield class實例，`__init__(self,matrixs)`會透過傳入的訓練資料matrixs進行網路權重W的訓練。



### \_\_init__()：
進行網路訓練，初始化鍵結值矩陣W
```python= 7
class Hopfield:
    w=[]
    theta = []
    def __init__(self,matrixs):
        print("start training")
        matrixs = np.array(matrixs)
        n = len(matrixs)
        self.j_row = len(matrixs[0])
        self.j_column = len(matrixs[0][0])
        self.p = self.j_row*self.j_column
        print(f"array shape = {self.j_row},{self.j_column}")

        print(f"len = {self.p}")
        self.w = np.zeros((self.p, self.p))
        for matrix in matrixs:
            print(np.array(matrix))
            matrix =np.array([matrix.flatten()]).T
            #print("flatten = \n",matrix)
            #print(f"transpose : \n{matrix.T}")
            self.w += np.dot(matrix ,matrix.T)

        self.w = self.w/self.p - n/self.p*np.eye(self.p)
        print(self.w)
        self.theta = np.array([np.sum(self.w, axis=1)]).T
        #print("theta:",self.theta)
        print("traning complete")
```

`row: 8~9`    首先紀錄輸入陣列的shape為`self.j_row`、`self.j_column`。
`row: 10`    計算鍵結矩陣W（大小為p \* p）的p。
`row: 14`    初始化鍵結值矩陣
`row: 15`    迴圈對每個訓練資料中的圖片矩陣`matrix`進行計算。
`row: 17`    將matrix flatten為`p * 1`的矩陣
`row: 20`    將`self.w`加上flattened matrix與自己transpose結果的內積（dot）
`row: 22`    根據公式將鍵結值矩陣初始化完成:
>將`所有內積相加結果/p`，減去`n(要記憶的圖形總數) / p * I(單位矩陣)`。
>![image](https://hackmd.io/_uploads/B1tzRlNNJg.png)

`row 24`    將`theta`初始化為`self.w`的`row sums`。
>![image](https://hackmd.io/_uploads/rkUJxbVE1x.png)




### predict_sync()：
使用同步更新的方式進行回想recall

```python=35
def predict_sync(self,matrix,ax = None,canvas = None,status_panel = None):
        print("start testing(sync).")
        last_result = [-1]
        result = []
        matrix = self.array_flatten(matrix)
        while not np.array_equal(last_result,result):
            self.recall_status_panel_update(status_panel)  
            last_result = result.copy()
            result = []
            
            result = np.dot(self.w,matrix)
            #result = result - self.theta
            for j,row in enumerate(result):
                result[j][0] =self.sign(row[0],matrix[j][0])
        
            self.refresh_picture(result.reshape(self.j_row,self.j_column),ax,canvas)
            matrix = result.copy()
        
        self.recall_status_panel_update(status_panel,done=True)               
        print("testing complete(sync).")
        return result.reshape(self.j_row,self.j_column)

```
`row 39`：`self.array_flatten`會先將輸入`matrix`轉換為`np.array`，並flatten為`p * 1`的矩陣。
`row 40`：while主迴圈檢查若`上次輸出結果`不等於`這次輸出結果`（代表輸出上為收斂）就持續進行更新直至收斂。

`row 42`：紀錄上一次輸出數值。
`row 45`：將權重與輸入相乘(一次算出每個神經元輸出)。
`row 46`：（可選）若選擇將theta設為0則可不用將result減去theta。
`row 47`：檢查輸出陣列的每個元素進行sign運算。
`row 51`：將新的輸入matix設為此次的輸出結果並回授給網路重複進行下一次的運算

重複以上流程直至輸出收斂（`上次輸出結果`等於`這次輸出結果`，輸出不再改變）為止。




### predict_async()：
使用非同步更新的方式進行回想recall


```python=62
def predict_async(self,matrix,max_epoch = 10,ax = None,canvas = None,status_panel = None):
        print("start testing(async).")
        matrix = self.array_flatten(matrix)
        last_result = []
        #print(matrix)
        for epoch in range(max_epoch):
            self.recall_status_panel_update(status_panel,epoch)
            if np.array_equal(matrix,last_result):
                max_epoch = epoch
                break
            last_result = matrix.copy()
            print(f"epoch : {epoch}")
            for j in range(self.p):
                y = 0
                # for i in range(self.p):
                #     y += self.w[j][i] * matrix[i][0]
                y = np.dot(self.w[j],matrix)
                #y -= self.theta[j][0]
                matrix[j][0] = self.sign(y,matrix[j][0])
                self.refresh_picture(matrix.reshape(self.j_row,self.j_column),ax,canvas)
            
        print("testing complete(async).")
        self.recall_status_panel_update(status_panel,max_epoch,True)
        self.refresh_picture(matrix.reshape(self.j_row,self.j_column),ax,canvas)
        return matrix.reshape(self.j_row,self.j_column)
```
`row 64` ：`self.array_flatten`會先將輸入`matrix`轉換為`np.array`，並flatten為`p * 1`的矩陣。
`row 69`：迴圈終止條件，如果新的回想結果與上次完全相同，代表回想已經收斂->跳出主迴圈提早結束。

`row 72`：紀錄上一次回想輸出。
`row 74~78`：每次更新回想結果`matrix`中的一個元素(透過鍵結值矩陣`self.w`中的一個row與輸入`matrix`的內積計算)

`row 79`：（可選）若選擇將theta設為0則可不用將result減去theta。
`row 80`：將內積的計算結果經過`sign`函數並更新至`matrix`中。

重複以上流程直至輸出收斂（`上次輸出結果`等於`這次輸出結果`，輸出不再改變）為止或到達`max_epoch`為止。



### 其他函式


:::spoiler
#### recall_status_panel_update()：
再訓練過程中更新使用者介面上顯示的recall status


```python=91
def recall_status_panel_update(self,status_panel,epoch = None,done = False):
    if status_panel:
        if not done:
            status_panel.set(f"\n回想中...\n")
        if done:
            status_panel.set(f"\n回想完成\n")

        if epoch!=None:
            status_panel.set(status_panel.get()+f"Epoch = {epoch}\n")
```
#### sign()：
可以改用兩層np.where來進行替代。
`result = np.where(arg > 0, 1, np.where(arg < 0, -1, arg))`
判斷輸入數值:
如果大於0輸出1
如果大於0輸出(傳入的參數2)
如果大於0輸出1

```python=102
def sign(self,value,last_value):
    if value > 0:
        return 1
    elif value == 0:
        return last_value
    else:
        return -1
```
#### array_flatten()：
將任意大小矩陣轉換成coulmn數為1的矩陣 `ex:3*3矩陣轉換為9*1矩陣`


```python=110
def array_flatten(self,matrix):
    matrix = np.array(matrix)
    matrix = np.array([matrix.flatten()])
    matrix = matrix.T
    return matrix

```
#### refresh_picture()：
更新使用者介面上的圖表。



```python=116
def refresh_picture(self,result,ax,canvas):
    if (ax!=None) and (canvas!=None):
        #print("refresh!")
        ax.cla()
        ax.imshow(result, cmap='gray', interpolation='nearest')
        canvas.draw()
        canvas.flush_events()
        #time.sleep(0.05)
```

:::






## main.py(介面部分)
實作使用者介面
:::spoiler 程式碼
```python= 1
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

```
:::



### training():
建立一個新的Hopfield實例，傳入參數matrixs_training進行網路訓練
```python=101
#訓練按鈕
def training():
    global Hopfield_netowrk
    Training_status.set("訓練中...")
    Hopfield_netowrk = Hopfield(matrixs_training)
    Training_status.set("訓練完成")

```

### testing_sync():
對訓練時建立的Hopfield_network呼叫其中的predict_sync() funtion，傳入欲進行回想的測試資料。

```python=101
#同步更新模式回想
def testing_sync():
    if not Hopfield_netowrk:
            print("尚未進行訓練")
    else:
        Hopfield_netowrk.predict_sync(matrixs_testing[PIC_ID.get()-1],ax=ax2,canvas=canvas2,status_panel=Testing_status)

```

### testing_async():
對訓練時建立的Hopfield_network呼叫其中的predict_async() funtion，傳入欲進行回想的測試資料。
```python=101
#非同步更新模式回想
def testing_async():
    show_data(matrixs_training[PIC_ID.get()-1],ax,canvas)
    if not Hopfield_netowrk:
        print("尚未進行訓練")
    else:
        Hopfield_netowrk.predict_async(matrixs_testing[PIC_ID.get()-1],max_epoch= EP.get(),ax=ax2,canvas=canvas2,status_panel=Testing_status)
    # show_data(recalled_pattern,ax,canvas)

```

###  flip_elements():  加分題2(自行加入雜訊並進行回想)
`row: 134`按照傳入的比例ex:0.1，先計算總共要改變得像素數量(flip_nums)
`row: 136`再透過np.random.choice()隨機選取對應數量的像素(indices)
`row: 140`將選取的像素進行反轉(1->-1;-1->1)。

```python=127
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
```



### 其他函式

:::spoiler 其他函式
#### select()
#### select_predict()
#### read_data()
#### show_data()
#### load_picture()
:::



# 實驗結果
## Basic

|序號|正確答案|測試資料|回想結果 (async)|正確|回想結果 (sync)|正確|
|-|-|-|-|-|-|-|
|1|![image](https://hackmd.io/_uploads/H1_iLZNNyx.png)|![image](https://hackmd.io/_uploads/B1ETLWEVke.png)|![image](https://hackmd.io/_uploads/BJUZwb4VJx.png)|:o:|![image](https://hackmd.io/_uploads/BkMXD-NEJg.png)|:o:|
|2|![image](https://hackmd.io/_uploads/H1-rPWNVkl.png)|![image](https://hackmd.io/_uploads/r1qSDWE4yg.png)|![image](https://hackmd.io/_uploads/Hy8Dvb4Vkl.png)|:o:|![image](https://hackmd.io/_uploads/S12KvWV4kl.png)|:o:|
|3|![image](https://hackmd.io/_uploads/HkNC_b441l.png)|![image](https://hackmd.io/_uploads/BkJJt-EV1g.png)|![image](https://hackmd.io/_uploads/B1Mlt-EE1g.png)|:o:|![image](https://hackmd.io/_uploads/BJg-t-E4ye.png)|:o:|

在Basic資料集中，不論是async或sync網路皆能完全回想正確。
所有回想皆能夠在總共使用2個epoch就完成。

# 加分題項目
## Bonus
|序號|正確答案|測試資料|回想結果 (async)|正確|回想結果 (sync)|正確|
|-|-|-|-|-|-|-|
|1|![image](https://hackmd.io/_uploads/BkkGhbENkl.png)|![image](https://hackmd.io/_uploads/H1lm2WV4yx.png)|![image](https://hackmd.io/_uploads/ryE43WVNJg.png)|:o:|![image](https://hackmd.io/_uploads/S1XBnbN4yg.png)|:o:|
|2|![image](https://hackmd.io/_uploads/rJrI3WEN1g.png)|![image](https://hackmd.io/_uploads/H1zw3b4Vkg.png)|![image](https://hackmd.io/_uploads/By2Pn-4N1l.png)|:o:|![image](https://hackmd.io/_uploads/Bk_On-44kg.png)|:o:|
|3|![image](https://hackmd.io/_uploads/ByLTYWENyg.png)|![image](https://hackmd.io/_uploads/SyWAKb44kx.png)|![image](https://hackmd.io/_uploads/rkxg9WNNke.png)|:o:|![image](https://hackmd.io/_uploads/BJpZcZNEJx.png)|:o:|
|4|![image](https://hackmd.io/_uploads/ry9Y2bVEke.png)|![image](https://hackmd.io/_uploads/S1Lc2b441g.png)|![image](https://hackmd.io/_uploads/BJqihWE4Jg.png)|:o:|![image](https://hackmd.io/_uploads/B1S23-441e.png)|:o:|
|5|![image](https://hackmd.io/_uploads/BkXphW4NJe.png)|![image](https://hackmd.io/_uploads/HJ3T3WNVJe.png)|![image](https://hackmd.io/_uploads/r1p0h-VV1g.png)|:o:|![image](https://hackmd.io/_uploads/SyQlpZVNke.png)|:o:|
|6|![image](https://hackmd.io/_uploads/HJfZpbNNyl.png)|![image](https://hackmd.io/_uploads/r1pbTWVEkl.png)|![image](https://hackmd.io/_uploads/BkJ4T-E4Jl.png)|:x:|![image](https://hackmd.io/_uploads/BJZrpWVEJl.png)|:x:|
|7|![image](https://hackmd.io/_uploads/H1oe1M44Jg.png)|![image](https://hackmd.io/_uploads/HyZMkGNN1e.png)|![image](https://hackmd.io/_uploads/BJSbyfENyg.png)|:o:|![image](https://hackmd.io/_uploads/BJ9z1MEV1g.png)|:o:|
|8|![image](https://hackmd.io/_uploads/S17N1GE4kx.png)|![image](https://hackmd.io/_uploads/HJlSkzVNkg.png)|![image](https://hackmd.io/_uploads/H16UJzEVyg.png)|:x:|![image](https://hackmd.io/_uploads/H1pwJGV41l.png)|:x:|
|9|![image](https://hackmd.io/_uploads/BJA_gGVN1g.png)|![image](https://hackmd.io/_uploads/BJY8WMV41l.png)|![image](https://hackmd.io/_uploads/H1DwbzVVyx.png)|:o:|![image](https://hackmd.io/_uploads/rJ-dZzE4Jl.png)|:o:|
|10|![image](https://hackmd.io/_uploads/BkzKWzEEkg.png)|![image](https://hackmd.io/_uploads/SkjYbMVVke.png)|![image](https://hackmd.io/_uploads/ByOjWMNEkx.png)|:x:|![image](https://hackmd.io/_uploads/r1IhWG44Je.png)|:x:|
|11|![image](https://hackmd.io/_uploads/rJD2GfN41x.png)|![image](https://hackmd.io/_uploads/r1W1mMEN1e.png)|![image](https://hackmd.io/_uploads/rk-xmzN4kx.png)|:o:|![image](https://hackmd.io/_uploads/BJERMf4Nyx.png)|:o:|
|12|![image](https://hackmd.io/_uploads/SyO-mfNNyl.png)|![image](https://hackmd.io/_uploads/BkmGQfNN1g.png)|![image](https://hackmd.io/_uploads/H1NXXMNEJg.png)|:x:|![image](https://hackmd.io/_uploads/HyTXmfNEkl.png)|:x:|
|13|![image](https://hackmd.io/_uploads/BkyHQGN4kg.png)|![image](https://hackmd.io/_uploads/SJOrQM4N1x.png)|![image](https://hackmd.io/_uploads/HJOImzNN1g.png)|:o:|![image](https://hackmd.io/_uploads/B17PXfNNJe.png)|:o:|
|14|![image](https://hackmd.io/_uploads/rk7OXGN4kx.png)|![image](https://hackmd.io/_uploads/SkhOmGVVyg.png)|![image](https://hackmd.io/_uploads/S1ntXMVE1e.png)|:x:|![image](https://hackmd.io/_uploads/B1oqQzNV1g.png)|:x:|
|15|![image](https://hackmd.io/_uploads/SJ82Qz4NJl.png)|![image](https://hackmd.io/_uploads/HkR3mGEEye.png)|![image](https://hackmd.io/_uploads/SypTQGE41x.png)|:o:|![image](https://hackmd.io/_uploads/BkkJVfENJx.png)|:o:|



第一個碰到的回想錯誤是序號六的，不論是async或sync更新方法都會錯誤的聯想成序號1的圖片。

序號8的圖片也碰到回想錯誤的情況，不論是async或sync更新方法都回想成了序號2的pattern。

序號10的圖片不論是async或sync更新方法皆回想出了訓練資料中不存在的pattern。
可以觀察到使用Hebbian規則調整網路的鍵結值時，有可能會使得網路能量之局部極小值的數目會超過原先儲存的資料數目"偽造狀態(Spurious states)"的現象。



## 自行加入雜訊進行回想

觀察不同比例的隨機雜訊下的回想表現。

圖片A
|雜訊比例|雜訊圖形|回想結果 (async)|正確|回想結果 (sync)|正確|
|-|-|-|-|-|-|
|10%|![image](https://hackmd.io/_uploads/rystOMNVkg.png)|![image](https://hackmd.io/_uploads/BJjsdfVN1x.png)|:o:|![image](https://hackmd.io/_uploads/S1V6OfVVkg.png)|:o:|
|10%|![image](https://hackmd.io/_uploads/B1NgYGV4yg.png)|![image](https://hackmd.io/_uploads/Bk4-YzVE1x.png)|:o:|![image](https://hackmd.io/_uploads/SyT-KzVN1l.png)|:o:|
|20%|![image](https://hackmd.io/_uploads/rkNNFMVE1e.png)|![image](https://hackmd.io/_uploads/H1BrKMNNke.png)|:o:|![image](https://hackmd.io/_uploads/HJwLtGNVyl.png)|:o:|
|20%|![image](https://hackmd.io/_uploads/Hy7CYfVVkx.png)|![image](https://hackmd.io/_uploads/B15J9zN4Jg.png)|:o:|![image](https://hackmd.io/_uploads/rJdg9fNV1e.png)|:o:|
|30%|![image](https://hackmd.io/_uploads/SkmN5MVV1l.png)|![image](https://hackmd.io/_uploads/SJNr9MNE1l.png)|:o:|![image](https://hackmd.io/_uploads/BkNUqGVNke.png)|:o:|
|30%|![image](https://hackmd.io/_uploads/Syzbz7VE1x.png)|![image](https://hackmd.io/_uploads/HyY1GQVVye.png)|:o:|![image](https://hackmd.io/_uploads/Hy5xGmVNJl.png)|:o:|
|40%|![image](https://hackmd.io/_uploads/HJwYZmNEJx.png)|![image](https://hackmd.io/_uploads/Sy4DWX4E1g.png)|:o:|![image](https://hackmd.io/_uploads/Hy-dWQ441l.png)|:o:|

由於字母A的圖片與另外兩者C、L差異顯著，故將相當高比例的像素進行反轉模型都仍能成功進行回想。

圖片C
|測試|雜訊比例|雜訊圖形|回想結果 (async)|正確|回想結果 (sync)|正確|
|-|-|-|-|-|-|-|
|1|10%|![image](https://hackmd.io/_uploads/ByY1oGE4yl.png)|![image](https://hackmd.io/_uploads/ByTlizN4kx.png)|:o:|![image](https://hackmd.io/_uploads/ByZ-jGEEJe.png)|:o:|
|2|20%|![image](https://hackmd.io/_uploads/rJMUiGE4kl.png)|![image](https://hackmd.io/_uploads/S1itszN4kl.png)|:o:|![image](https://hackmd.io/_uploads/ByK5sfNVkg.png)|:o:|
|3|20%|![image](https://hackmd.io/_uploads/HymDhzVEJg.png)|![image](https://hackmd.io/_uploads/BJNu2MNNkl.png)|:o:|![image](https://hackmd.io/_uploads/HJkKhf44kl.png)|:o:|
|4|30%|![image](https://hackmd.io/_uploads/B1ET2fVNyx.png)|![image](https://hackmd.io/_uploads/HkJMpzVVye.png)|:x:|![image](https://hackmd.io/_uploads/rJbxTGV41g.png)|:x:|
|5|30%|![image](https://hackmd.io/_uploads/SkNS6zNN1e.png)|![image](https://hackmd.io/_uploads/HyJwTMNNkl.png)|:o:|![image](https://hackmd.io/_uploads/rkTPaGVVke.png)|:o:|
|6|40%|![image](https://hackmd.io/_uploads/BJ9NgQ4Nye.png)|![image](https://hackmd.io/_uploads/H1n8gmV4Jl.png)|:x:|![image](https://hackmd.io/_uploads/BJXHxmNN1x.png)|:x:|



可以看到因為C的圖像與L圖像較為相似，故在30%雜訊時，回想就有不穩定的情況了(但也尚有高機率回想正確)。在測試4中也碰到了"偽造狀態(Spurious states)"的現象。








# 實驗結果分析及討論


## 同步與非同步更新的比較:
在本次實驗中，回想測試結果皆是兩者正確或兩者皆不正確，沒有觀察到其中一方正確另一方錯誤的情況，但錯誤回想的時後兩者回想出來的結果有時不同，但同步更新的運算速度會明顯比非同步還要快一些(測試時有取消兩者的圖像更新，以避免因為更新次數不同所造成的影響)。

## 不同等級的雜訊回想結果
在"自行加入雜訊進行回想"的章節中可以觀察到，若一個pattern與訓練集中的其他pattern差異較大(ex:A與C或L)，該pattern就可以容忍較高的雜訊。

實驗結果:
A pattern在40%雜訊時仍可以有7成的完全正確回想機率。
而 pattern C因為與L圖形上較為相近在30%雜訊時就只剩7成正確回想機率了。


## 設定theta為鍵結值row sum，可能導致了過擬合(都回想成同一結果)

將theta設定為鍵結值矩陣的每個row sum，這樣設定下進行回想時，碰到大部分的輸入圖片最後都會回想收斂成同一張圖片的結果。

## 進階題中錯誤的結果:
進階題中相似pattern的回想:
進階題中錯誤的圖形序號有6、8、10、12、14

:::danger
|序號|正確答案|測試資料|回想結果 (async)|正確|回想結果 (sync)|正確|
|-|-|-|-|-|-|-|
|6|![image](https://hackmd.io/_uploads/HJfZpbNNyl.png)|![image](https://hackmd.io/_uploads/r1pbTWVEkl.png)|![image](https://hackmd.io/_uploads/BkJ4T-E4Jl.png)|:x:|![image](https://hackmd.io/_uploads/BJZrpWVEJl.png)|:x:|
|8|![image](https://hackmd.io/_uploads/S17N1GE4kx.png)|![image](https://hackmd.io/_uploads/HJlSkzVNkg.png)|![image](https://hackmd.io/_uploads/H16UJzEVyg.png)|:x:|![image](https://hackmd.io/_uploads/H1pwJGV41l.png)|:x:|
|10|![image](https://hackmd.io/_uploads/BkzKWzEEkg.png)|![image](https://hackmd.io/_uploads/SkjYbMVVke.png)|![image](https://hackmd.io/_uploads/ByOjWMNEkx.png)|:x:|![image](https://hackmd.io/_uploads/r1IhWG44Je.png)|:x:|
|12|![image](https://hackmd.io/_uploads/SyO-mfNNyl.png)|![image](https://hackmd.io/_uploads/BkmGQfNN1g.png)|![image](https://hackmd.io/_uploads/H1NXXMNEJg.png)|:x:|![image](https://hackmd.io/_uploads/HyTXmfNEkl.png)|:x:|
|14|![image](https://hackmd.io/_uploads/rk7OXGN4kx.png)|![image](https://hackmd.io/_uploads/SkhOmGVVyg.png)|![image](https://hackmd.io/_uploads/S1ntXMVE1e.png)|:x:|![image](https://hackmd.io/_uploads/B1oqQzNV1g.png)|:x:|
:::

可以觀察到:
### 測試6:
測試6的結果聯想到序號一圖形，推測可能原因是在訓練集中，序號一pattern的圖形有多份同樣的資料，導致對序號一圖形過擬合問題。
另外序號6圖形本身與序號1圖形相似度較高也可能是原因之一。
### 測試8:
測試8的回想中，模型錯誤的回想成了序號二的圖形，觀察訓練資料可以發現序號二的pattern與序號一pattern一樣，在訓練集中有多份同樣的資，可能是導致測試8錯誤的原因。

### 測試10:
測試10的結果中，不論是同步/非同步聯想最後都聯想出了訓練中不存在的pattern，也就是使用Hebbian規則調整網路的鍵結值時，可能會使得網路能量之局部極小值的數目會超過原先儲存的資料數目"偽造狀態(Spurious states)"的現象。

### 測試12、14:
在這兩個測試中，網路都將這些圖形回想成了訓練資料中相似的其他pattern。
可以觀察到若訓練集中有相似的圖片，也可能導致模型在進行這些圖片的回想時錯誤回想成另外一種。




