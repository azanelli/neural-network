#
# nn makefile
#

TARGETDIR = bin
NN = ${TARGETDIR}/nn
TARGETS = $(NN)

CC = g++
CPPFLAGS = -W -Wall
MKDIR = mkdir -p
CP = cp
RM = rm -rf

all: $(TARGETS)

$(NN): nn.o nntraining.o nntest.o trainer.o tester.o backpropagation.o \
       neuralnetwork.o dataset.o unit.o global.o
	$(MKDIR) $(TARGETDIR)/
	$(CC) $(CPPFLAGS) nn.o nntraining.o nntest.o trainer.o tester.o \
	    backpropagation.o neuralnetwork.o dataset.o unit.o global.o -o $(NN)
	$(CP) help.txt $(TARGETDIR)/

nn.o: nn.cpp nntraining.h nntest.h global.h
	$(CC) $(CPPFLAGS) -c nn.cpp

nntraining.o: nntraining.h nntraining.cpp neuralnetwork.h backpropagation.h \
              trainer.h global.h exception.h
	$(CC) $(CPPFLAGS) -c nntraining.cpp

nntest.o: nntest.h nntest.cpp neuralnetwork.h tester.h global.h exception.h
	$(CC) $(CPPFLAGS) -c nntest.cpp

trainer.o: trainer.h trainer.cpp backpropagation.h neuralnetwork.h dataset.h \
           global.h exception.h
	$(CC) $(CPPFLAGS) -c trainer.cpp

tester.o: tester.h tester.cpp neuralnetwork.h dataset.h global.h exception.h
	$(CC) $(CPPFLAGS) -c tester.cpp

backpropagation.o: backpropagation.h backpropagation.cpp neuralnetwork.h \
                   global.h
	$(CC) $(CPPFLAGS) -c backpropagation.cpp

neuralnetwork.o: neuralnetwork.h neuralnetwork.cpp unit.h global.h exception.h
	$(CC) $(CPPFLAGS) -c neuralnetwork.cpp

dataset.o: dataset.h dataset.cpp global.h exception.h
	$(CC) $(CPPFLAGS) -c dataset.cpp

unit.o: unit.h unit.cpp global.h exception.h
	$(CC) $(CPPFLAGS) -c unit.cpp

global.o: global.h global.cpp exception.h
	$(CC) $(CPPFLAGS) -c global.cpp

clean:
	$(RM) *.o $(TARGETS) $(DIRDOC) $(TARGETDIR)

