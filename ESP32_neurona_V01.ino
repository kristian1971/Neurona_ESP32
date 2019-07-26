#include <Neurona.h>

#define NET_INPUTS 9
#define NET_HIDDEN_1 2
#define NET_HIDDEN_2 2
#define NET_OUTPUTS 2
#define NET_LAYERS 3

int counter1 = 0;
int counter2 = 0;
int sum_all = 0;
int counter3 = 0;
int layerSizes[] = {NET_HIDDEN_1,NET_HIDDEN_2,NET_OUTPUTS, -1};
double const initW[] = {1.7025086333947348,-1.4041139239275011,1.9008654649054932,-0.29872782302775536,1.5692774474001,-1.4346057251451105,1.5151434584310308,-0.40548709541632627,1.8323001615697159,-0.6593753501925649,1.3701243257825757,-0.4527907160061316,1.3188511840889872,-0.6093040881381074,1.4604941453541367,-1.4045611444635873,1.6508505466756929,-1.0227764701531183,0.9291265817231106,-0.023770882539236034,2.7443607324187718,3.5850644199506267,2.407410424693586,-2.6604688148327185,-3.4415009773507403,-2.3819970728360462};

MLP mlp(NET_INPUTS,NET_OUTPUTS,layerSizes,MLP::LOGISTIC,initW,false);

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();
}

void loop() {
  // put your main code here, to run repeatedly:
counter3=counter3+1;
counter1=micros();
double instance[] = {0.118,0.878,0.392,0.737,0.137,0.886,0.106,0.690,0.200};
double *out = mlp.forward(instance);
int index = mlp.getActivation(instance);
counter2=micros();
sum_all = sum_all + (counter2-counter1);
Serial.printf("INDEX %i\n", index); 
Serial.printf("MICROS %i\n", sum_all/counter3); 
Serial.printf("COUNTER %i\n", counter3); 
}
