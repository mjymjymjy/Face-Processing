# -*- coding:UTF-8 -*-
from changeface import ChangeFace
from Beauty import Beauty
print("This is a simple face exchange or beauty program. Please select the desired function:")
print("C/c represents the face exchange function, and B/b represents the beauty function.")
print("Please enter Q/q to exit.")
while(1):
    inp = input("Please input:")
    if inp=='C' or inp=='c':
        pic1_path = input("Please enter the path of the first picture: ")
        pic2_path = input("Please enter the path of the second picture: ")
        pic1 = pic1_path
        pic2 = pic2_path
        ChangeFace = ChangeFace(pic1,pic2)
        ChangeFace.main()
        print('ChangeFace success!')

    elif inp=='B' or inp=='b':
        pic_path = input("Please enter the path of the picture: ")
        pic = pic_path
        Beauty = Beauty(pic)
        Beauty.main()
        print('Bueaty success!')
    elif inp == 'q' or inp == 'Q':
        break
    elif inp != 'C' or inp != 'c' or inp != 'B' or inp != 'b' or inp != 'q' or inp != 'Q':
        print("Input error, please check and input the correct letter!")
