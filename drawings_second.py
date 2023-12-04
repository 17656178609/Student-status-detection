import turtle

# 设置画布
screen = turtle.Screen()
screen.bgcolor("white")

# 创建一个画笔
pen = turtle.Turtle()
pen.shape("turtle")
pen.speed(1)
pen.color("red")

# 绘制爱心
pen.begin_fill()
pen.left(140)
pen.forward(224)
for _ in range(200):
    pen.right(1)
    pen.forward(2)
pen.left(120)
for _ in range(200):
    pen.right(1)
    pen.forward(2)
pen.forward(224)
pen.end_fill()

# 隐藏画笔
pen.hideturtle()

# 保持窗口打开
turtle.done()
