# -*- coding: utf-8 -*-


from tkinter import *
from PIL import Image, ImageTk
import tkinter.filedialog
from predict import *
import config
import math

IMAGE_LABELS = []
REG_TEXT_LABELS = []

IMAGE_SHOW_SIZE = 100
IMAGE_ROWS = 4
IMAGE_COLUMNS = 4
MNIST_IMAGE_SIZE = 28

root = tkinter.Tk()
#root.update()        手工刷新
filenames = None
current_page = 1
total_pages = 1
recognized = False
result=None

#######################################
#  事先创建所有的image labels
#######################################
def create_controls():
    for i in range(2, IMAGE_ROWS+2):
        for j in range(2, IMAGE_COLUMNS+2):
            load = Image.open('./handwrite_digits_pics/empty.png')
            render = ImageTk.PhotoImage(load)
            # Label这里的width和height单位会因这个label的用途而变化，如果是作为image，则单位是像素，如果是text，则单位为text units
            img_label = tkinter.Label(root,image=render, width=IMAGE_SHOW_SIZE, height=IMAGE_SHOW_SIZE)
            img_label.grid(row=i, column=2*j, sticky=W, padx=1, pady=1, ipadx=0, ipady=1)
            IMAGE_LABELS.append(img_label)

            reg_text_label = Label(root, text='',width=1, height=1,fg='blue',font='Helvetica -30 bold')
            reg_text_label.grid(row=i, column=2*j+1,sticky=W, padx=1, pady=1, ipadx=0, ipady=1)
            REG_TEXT_LABELS.append(reg_text_label)

#######################################
#  选择要识别的图片
#######################################
def chooseImages():
    global filenames
    filenames = tkinter.filedialog.askopenfilenames()
    print(filenames)
    if filenames == '':
        lb.config(text="您没有选择任何文件")
        return
    num_per_page = IMAGE_ROWS * IMAGE_COLUMNS
    global total_pages
    if (len(filenames)%num_per_page == 0):
        total_pages = len(filenames) //num_per_page
    else:
        total_pages = len(filenames) // num_per_page + 1

    global recognized
    global current_page
    global result
    recognized = False
    current_page = 1
    result = None

    showImages()
    showResults()

#######################################
#  按照页码显示图片
#######################################
def showImages():
    global filenames
    global current_page
    page_label.config(text="{0}/{1}".format(current_page, total_pages))
    file_nums = len(filenames)
    num_per_page = IMAGE_ROWS * IMAGE_COLUMNS
    start_index, end_index = get_start_end_index(file_nums, num_per_page, current_page)
    image_label_index = 0
    for file_index in range(start_index, end_index):
        file_name = filenames[file_index]
        img = Image.open(file_name)
        img = img.resize((IMAGE_SHOW_SIZE, IMAGE_SHOW_SIZE), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(img)

        IMAGE_LABELS[image_label_index].config(image=render)
        IMAGE_LABELS[image_label_index].image=render
        image_label_index += 1
    print("image_label_index=",image_label_index)
    print("num_per_page=",num_per_page)
    if image_label_index < num_per_page:
        for i in range(image_label_index, num_per_page):
            #print(i)
            load = Image.open('./handwrite_digits_pics/empty.png')
            render = ImageTk.PhotoImage(load)
            IMAGE_LABELS[i].config(image=render)
            IMAGE_LABELS[i].image = render
    root.update()

#######################################
#  按照页码显示识别结果
#######################################
def showResults():
    global filenames
    global current_page
    global result

    num_per_page = IMAGE_ROWS * IMAGE_COLUMNS
    if result == None:
        for i in range(num_per_page):
            REG_TEXT_LABELS[i].config(text='')
        return

    file_nums = len(filenames)
    start_index, end_index = get_start_end_index(file_nums, num_per_page, current_page)
    reg_label_index = 0
    for file_index in range(start_index, end_index):
        REG_TEXT_LABELS[reg_label_index].config(text=result[file_index] )
        reg_label_index += 1

    if reg_label_index < num_per_page:
        for i in range(reg_label_index, num_per_page):
            REG_TEXT_LABELS[i].config(text='')
    root.update()


def get_start_end_index(file_nums, num_per_page, current_page):
    start_index = (current_page-1)*num_per_page
    end_index = current_page*num_per_page
    if file_nums < end_index:
        end_index = file_nums
    return start_index, end_index

#####################################
# 识别
#####################################
def recognize():
    global filenames
    global recognized
    global is_mnist
    recognized = True
    if filenames == None:
        lb.config(text="您还没有选择任何文件")
        return
    images = []
    for filename in filenames:
        if is_mnist.get() == False:  # uncheked
            print("unchecked!")
            data = process_pic(filename, mnist_pic=False)
        else:
            print("checked!")
            data = process_pic(filename, mnist_pic=True)
        print('data shape:',data.shape)
        images.append(data)
    images = np.array(images)
    images = images.reshape(len(filenames),MNIST_IMAGE_SIZE*MNIST_IMAGE_SIZE)  # 必须！
    images = images.astype('float32') # 必须，否则会报错说类型不匹配，因为模型里头的参数用的是float32类型，这里不转换默认是double
    global result
    result = do_inference(config.server, config.work_dir,
                           config.concurrency, images, None)
    print(result)
    showResults()

def previousPage():
    global recognized
    global current_page
    if current_page == 1:
        pass
    else:
        current_page -= 1
    if recognized:
        showResults()
    showImages()

def nextPage():
    global recognized
    global current_page
    global total_pages
    if current_page == total_pages:
        pass
    else:
        current_page += 1
    if recognized:
        showResults()
    showImages()

create_controls()
btn = Button(root,text="选择要识别的图片",command=chooseImages)
btn.grid(row=0,column=0, columnspan=3, sticky=W, padx=1, pady=1, ipadx=0, ipady=1)

btn = Button(root,text="识别",command=recognize,fg='blue')
btn.grid(row=1,column=0, columnspan=1, sticky=W, padx=1, pady=1, ipadx=0, ipady=1)

is_mnist = IntVar()
ck_btn=Checkbutton(root, text="mnist 数据集", variable=is_mnist)
ck_btn.grid(row=1, column=1,columnspan=2)

btn = Button(root,text="Prev",command=previousPage)
btn.grid(row=2,column=0, sticky=W, padx=1, pady=1, ipadx=0, ipady=1)

page_label = Label(root,text = '0/0', width=3)
page_label.grid(row=2,column=1, padx=1, pady=1, ipadx=0, ipady=1)

btn = Button(root,text="Next",command=nextPage)
btn.grid(row=2,column=2, sticky=W,padx=1, pady=1, ipadx=0, ipady=1)



# 进入消息循环
root.mainloop()