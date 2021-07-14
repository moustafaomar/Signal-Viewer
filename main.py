import sys
import random
import matplotlib
matplotlib.use("Qt5Agg")
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLineEdit,QComboBox, QFileDialog, QSlider, QGroupBox, QApplication, QMainWindow, QMenu, QHBoxLayout , QVBoxLayout, QSizePolicy, QMessageBox, QWidget,QPushButton
from numpy import arange, sin, pi,fft
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from matplotlib.widgets import SpanSelector
from functools import partial

fmax=250
fs=500
period=1/500
class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100,file=None):
        self.file = file
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class MyStaticMplCanvas(MyMplCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100,file=None):
        super(). __init__(parent=None, width=5, height=4, dpi=100,file=None)
        self.cmap = 'viridis'
    def addSpectrogram(self,lineData,min=None,max=None):
        NFFT = 1024
        self.axes.cla()
        self.axes.specgram(lineData, NFFT=NFFT, Fs=fs, noverlap=900,cmap=self.cmap)
        self.draw()
        try:
            self.axes.set_ylim(min,max)
            self.draw()
        except:
            pass

    def setCmap(self,text):
        self.cmap = text

class MyDynamicMplCanvas(MyMplCanvas):
    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        self.lines = self.file.readlines()
        self.y = [float(line.split(',')[0]) for line in self.lines]
        x = np.arange(len(self.y))
        self.axes.set_xlim(0,100)
        self.speed = 1000
        self.axes.plot(x,self.y, 'r')
        self.draw()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_figure)
        self.timer.start(self.speed)
    def Equalizer(self,values):
        main_file=self.y
        fourierTransform = (np.fft.rfft(main_file))
        numberOfPoints = len(main_file)
        x_frequency = np.linspace(0.0, 1.0/(2.0*period), numberOfPoints//2)
        new_frequency=[]
        for slider_counter in range(0,10):
            new_frequency.append([i for i, n in enumerate(x_frequency) if n>=fmax*(slider_counter)/10 and n<=fmax*(slider_counter+1)/10 ])
        slider_number=0
        for frequency in new_frequency:
          fourierTransform[frequency] = fourierTransform[frequency]*values[slider_number]
          slider_number+=1
        Equalizer_result = np.fft.irfft(fourierTransform) 
        Equalizer_result = np.round(Equalizer_result,2)
        x_equalizer = arange(len(Equalizer_result))
        l,r = self.axes.set_xlim()
        self.axes.cla()
        self.axes.plot(x_equalizer,Equalizer_result)
        self.axes.set_xlim(l,r)
        self.draw()
    def spectrogramRange(self,min,max,speclines):
        numberOfPoints = len(speclines)
        yf=np.fft.rfft(speclines)
        xf = np.linspace(0.0, 1.0/(2.0*period), numberOfPoints//2)
        z=[]
        newyf=[]
        z.append([i for i, n in enumerate(xf) if n>=min[0] and n<=max[0] ])
        for array in z:
            for i in array:
                newyf.append(yf[i])
        return np.fft.irfft(newyf)
    def scrollleft(self):
        self.scroll_helper(-10)
    def scrollright(self):
        self.scroll_helper(10)
    def zoomin(self):
        self.zoom_helper(0)
    def zoomout(self):
        self.zoom_helper(1)
    def scroll_helper(self,number):
        #Number is the number to be added to both limits
        left, right = self.axes.set_xlim()
        if number< 0 and left+number < 0 or right+number>len(self.y) and number>0:
            return
        left+=number
        right+=number
        left, right = self.axes.set_xlim(left,right)
        self.draw()
    def zoom_helper(self,identifier):
        #Identifier 0 for zoomin, 1 for zoomout
        left, right = self.axes.set_xlim()
        average = (left+right)/2
        left_for_calculation = average if identifier == 0 else left
        right_for_calculation = average if identifier ==0 else right
        left_new = left_for_calculation-abs((average - left)/2)
        right_new = right_for_calculation+abs((average - right)/2)
        left,right = self.axes.set_xlim(left_new,right_new)
        if left_new < 0 or right_new > len(self.y):
            return
        self.draw()
        
    def update_figure(self):
        left, right = self.axes.set_xlim()
        if(right+1 > len(self.y)):
            return
        self.axes.set_xlim(left+100,right+100)
        self.draw()
    def pause(self):
        self.timer.stop()
    def play(self):
        self.timer.start(self.speed)
    def getY(self):
        return self.axes.get_lines()[0].get_ydata()
    def setSpeed(self,speed):
        self.speed = speed
        self.timer.start(self.speed)
class ApplicationWindow(QMainWindow):
    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearLayout(item.layout())
    def __init__(self):
        QMainWindow.__init__(self)
        self.children = []
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Signal Viewer")
        self.resize(900, 800)
        self.file_menu = QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)
        self.actionOpen_file = QtWidgets.QAction(self)
        self.actionOpen_file.setObjectName("actionOpen_file")
        self.actionOpen_file.setText("Open file")
        self.actionOpen_file.setShortcut("Ctrl+O")
        self.actionOpen_file.triggered.connect(self.OpenFile)
        self.actionNew_file = QtWidgets.QAction(self)
        self.actionNew_file.setObjectName("actionNew_file")
        self.file_menu.addAction(self.actionOpen_file)
        self.file_menu.addAction(self.actionNew_file)
        self.actionNew_file.setText("New file")
        self.actionNew_file.setShortcut("Ctrl+N")
        self.actionNew_file.triggered.connect(self.NewFile)
        self.main_widget = QWidget(self)
        self.actionReport = QtWidgets.QAction(self)
        self.actionReport.setObjectName("actionReport")
        self.actionReport.setText("Report")
        self.actionReport.setShortcut("Ctrl+R")
        self.actionReport.triggered.connect(self.GenerateReport)
        self.file_menu.addAction(self.actionReport)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        self.layout = QHBoxLayout(self.main_widget)
    def GenerateReport(self):
        filenames = QFileDialog().getOpenFileNames(options=QFileDialog.DontUseNativeDialog)
        i = 0
        fig, axs = plt.subplots(len(filenames[0]),2)
        for filename in filenames[0]:
            file = open(filename)
            datalines = file.readlines()
            y = [float(line.split(',')[0]) for line in datalines]
            x = np.arange(len(y))
            axs[i, 0].plot(x, y)
            dt = 0.0005
            NFFT = 1024
            Fs = int(1.0 / dt)
            Pxx, freqs, bins, im = axs[i, 1].axes.specgram(y, NFFT=NFFT, Fs=Fs, noverlap=900)
            i+=1
        plt.savefig('report.pdf')  




    def drawButtons(self):
        self.pauseButton = QPushButton(self)
        self.pauseButton.setText("Pause")          #text
        self.pauseButton.setShortcut('Ctrl+D')  #shortcut key
        self.pauseButton.clicked.connect(self.pause)
        self.pauseButton.setToolTip("Close the widget") #Tool tip
        self.playButton = QPushButton(self)
        self.playButton.setText("Play")          #text
        self.playButton.setShortcut('Ctrl+P')  #shortcut key
        self.playButton.setToolTip("Close the widget") #Tool tip
        self.playButton.clicked.connect(self.play)
        self.Zoom = QPushButton(self)
        self.Zoom.setText("Zoom In")          #text
        self.Zoom.setShortcut('Ctrl+Z')  #shortcut key
        self.Zoom.setToolTip("Close the widget") #Tool tip
        self.Zoom.clicked.connect(self.zoomin)
        self.ZoomOut = QPushButton(self)
        self.ZoomOut.setText("Zoom Out")          #text
        self.ZoomOut.setShortcut('Ctrl+Z')  #shortcut key
        self.ZoomOut.setToolTip("Close the widget") #Tool tip
        self.ZoomOut.clicked.connect(self.zoomout)
        self.scrollLeft = QPushButton(self)
        self.scrollLeft.setText("Scroll Left")          #text
        self.scrollLeft.setShortcut('Ctrl+L')  #shortcut key
        self.scrollLeft.setToolTip("Close the widget") #Tool tip
        self.scrollLeft.clicked.connect(self.scrollleft)
        self.scrollRight = QPushButton(self)
        self.scrollRight.setText("Scroll Right")          #text
        self.scrollRight.setShortcut('Ctrl+R')  #shortcut key
        self.scrollRight.setToolTip("Close the widget") #Tool tip
        self.scrollRight.clicked.connect(self.scrollright)
    def drawSliders(self):
        self.sliders = []
        for slider_number in range(0,10):
            self.sliders.append(QSlider(QtCore.Qt.Vertical))
            self.sliders[slider_number].setFocusPolicy(QtCore.Qt.NoFocus)
            self.sliders[slider_number].setRange(0,100)
            self.sliders[slider_number].setMinimum(0)
            self.sliders[slider_number].setMaximum(5)
            self.sliders[slider_number].setSingleStep(1)
            self.sliders[slider_number].setTickInterval(1)
            self.sliders[slider_number].setTickPosition(QSlider.TicksBelow)
            self.sliders[slider_number].setValue(1)
            self.sliders[slider_number].valueChanged.connect(self.Equalizer)
    def Equalizer(self):
        slider_values = [self.sliders[slider_number].value() for slider_number in range(0,10)]
        self.dc2.Equalizer(slider_values)
        self.sc.addSpectrogram(self.dc2.getY())

    def spectrogram_Sliders(self):
        slider_maxvalue = [self.MaxSlider.value()]
        slider_minvalue = [self.MinSlider.value()]
        resultSignal = self.dc2.spectrogramRange(slider_minvalue, slider_maxvalue,self.dc2.getY())
        self.sc.addSpectrogram(resultSignal,slider_minvalue[0],slider_maxvalue[0])
    def OpenFile(self):
        try:
            self.clearLayout(self.layout)
        except:
            pass
        self.drawButtons()
        self.drawSliders()
        filename = QFileDialog().getOpenFileName(options=QFileDialog.DontUseNativeDialog)
        try:
            f1 = open(filename[0])
        except:
            return
        f2 = open(filename[0])
        self.dc,self.dc2 = MyDynamicMplCanvas(self.main_widget, width=5, height=4, dpi=100,file=f1),MyDynamicMplCanvas(self.main_widget, width=5, height=4, dpi=100,file=f2)
        self.sc = MyStaticMplCanvas(self.main_widget, width=4, height=3, dpi=100,file=f1)
        self.sc.addSpectrogram(self.dc2.getY())
        left = QVBoxLayout()
        buttons = QHBoxLayout()
        buttons.addWidget(self.pauseButton)
        buttons.addWidget(self.playButton)
        buttons.addWidget(self.Zoom)
        buttons.addWidget(self.ZoomOut)
        buttons.addWidget(self.scrollLeft)
        buttons.addWidget(self.scrollRight)
        combo = QComboBox(self)
        combo.addItem("viridis")
        combo.addItem("plasma")
        combo.addItem("inferno")
        combo.addItem("magma")
        combo.addItem("cividis")
        combo.activated[str].connect(self.onChanged)
        textbox = QLineEdit(self)
        textbox.setFixedSize(50,20)
        textbox.setText(str(1000))
        textbox.textChanged[str].connect(self.speedChanged)
        buttons.addWidget(combo)
        buttons.addWidget(textbox)
        horizontalGroupBox = QGroupBox("Controls")
        horizontalGroupBox.setLayout(buttons)
        SlidersgroupBox = QGroupBox("Equalizer")
        sliders = QHBoxLayout()
        for slider_number in range(0,10):
            sliders.addWidget(self.sliders[slider_number])
        SlidersgroupBox.setLayout(sliders)
        SpectrogramSliders = QGroupBox("Spectrogram Sliders")
        self.MaxSlider = QSlider(QtCore.Qt.Vertical)
        self.MaxSlider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.MaxSlider.setRange(0,250)
        self.MaxSlider.setSingleStep(50)
        self.MaxSlider.setTickInterval(50)
        self.MaxSlider.setTickPosition(QSlider.TicksBelow)
        self.MaxSlider.setValue(250)
        self.MinSlider = QSlider(QtCore.Qt.Vertical)
        self.MinSlider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.MinSlider.setRange(0,250)
        self.MinSlider.setSingleStep(50)
        self.MinSlider.setTickInterval(50)
        self.MinSlider.setTickPosition(QSlider.TicksBelow)
        self.MinSlider.setValue(0)
        MinMaxSliders= QHBoxLayout()
        MinMaxSliders.addWidget(self.MinSlider)
        MinMaxSliders.addWidget(self.MaxSlider)
        self.MinSlider.valueChanged.connect(self.spectrogram_Sliders)
        self.MaxSlider.valueChanged.connect(self.spectrogram_Sliders)
        SpectrogramSliders.setLayout(MinMaxSliders)
        left.addWidget(horizontalGroupBox)
        left.addWidget(self.dc)
        left.addWidget(self.dc2)
        left.addWidget(self.sc)
        self.layout.addLayout(left)
        self.layout.addWidget(SlidersgroupBox)
        self.layout.addWidget(SpectrogramSliders)
    def onChanged(self,text):
        self.sc.setCmap(text)
        self.sc.addSpectrogram(self.dc2.getY())
    def speedChanged(self,speed):
        self.dc.setSpeed(int(speed))
        self.dc2.setSpeed(int(speed))
    def zoomin(self):
        self.dc.zoomin(),self.dc2.zoomin()
    def zoomout(self):
        self.dc.zoomout(),self.dc2.zoomout()
    def scrollleft(self):
        self.dc.scrollleft(),self.dc2.scrollleft()
    def scrollright(self):
        self.dc.scrollright(),self.dc2.scrollright()
    def fileQuit(self):
        self.close()
    def pause(self):
        self.dc.pause(),self.dc2.pause()
    def play(self):
        self.dc.play(),self.dc2.play()
    def closeEvent(self, ce):
        self.fileQuit()
    def NewFile(self):
        self.newWindow = ApplicationWindow()
        self.children.append(self.newWindow)
        self.newWindow.setWindowTitle("Signal Viewer")
        self.newWindow.show()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.setWindowTitle("Signal Viewer")
    aw.show()
    app.exec_()