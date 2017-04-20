import com.sun.org.apache.xpath.internal.SourceTree;

import java.io.*;
import java.util.*;

public class Perceptrone_model {

    double weights[][][]; //веса сети; 1 компонента - номер ребер между i и i+1 cлоем; 2 компонента - i+i слой; 3 компонента - i-тый слой;
    double layers[][]; //значения нейронов; 1 - слой; 2 - номер нейрона;
    double output_of_layers[][]; //выходы нейронов; 1 - слой; 2 - номер нейрона;
    double biases[][]; //смещения; 1 - слой; 2 - номер нейрона;
    double errors[][]; //ошибки в нейронах; используется для алгоритма обратного распространения ошибки;
    double gradient_of_weights[][][]; //градиент для весов;
    int kol_of_layers; //количество слоев в нейронной сети;
    double gradient_of_biases[][]; //градиент для смещений;
    double learn_rate=0.1;//cкорость обучения;
    int mini_batch=2; //количество элементов для стохастичего градиента;
    int number_of_epoch=2000;
    double previous_weights[][][];//веса для использования в обучении с помощью момента
    double Lambda=0.5;//коэффициент момента

    //конструктор; 1 параметр - количество слоев; 2 параметр - количество нейронов в каждом слое;
    public Perceptrone_model(int kol_layers, int... number_of_neurons_in_layer) {

        Random rand = new Random();

        if (kol_layers < 2) System.exit(0);
        layers = new double[kol_layers][];
        kol_of_layers = kol_layers;
        output_of_layers = new double[kol_layers][];
        biases = new double[kol_layers][];
        weights = new double[kol_layers - 1][][];
        errors = new double[kol_layers][];
        gradient_of_weights = new double[kol_layers - 1][][];
        gradient_of_biases = new double[kol_layers][];
        previous_weights=new double[kol_layers-1][][];

        for (int i = 0; i < kol_layers; i++) {
            layers[i] = new double[number_of_neurons_in_layer[i]];
            output_of_layers[i] = new double[number_of_neurons_in_layer[i]];
            biases[i] = new double[number_of_neurons_in_layer[i]];
            gradient_of_biases[i] = new double[number_of_neurons_in_layer[i]];
            errors[i] = new double[number_of_neurons_in_layer[i]];
            for (int j = 0; j < biases[i].length; j++) {
                if (i != 0) biases[i][j] = rand.nextGaussian();
                else biases[i][j] = 0;
                gradient_of_biases[i][j]=0;
            }
            if (i != kol_layers - 1) {
                weights[i] = new double[number_of_neurons_in_layer[i + 1]][];
                previous_weights[i] = new double[number_of_neurons_in_layer[i + 1]][];
                gradient_of_weights[i] = new double[number_of_neurons_in_layer[i + 1]][];
                for (int j = 0; j < weights[i].length; j++) {
                    weights[i][j] = new double[number_of_neurons_in_layer[i]];
                    previous_weights[i][j]=new double[number_of_neurons_in_layer[i]];
                    gradient_of_weights[i][j] = new double[number_of_neurons_in_layer[i]];
                    for (int z = 0; z < number_of_neurons_in_layer[i]; z++) {
                        weights[i][j][z] = rand.nextGaussian();
                        previous_weights[i][j][z]=0;
                        gradient_of_weights[i][j][z]=0;
                    }
                }
            }
        }
    }

    //инициализация весов 1 слоя;
    public void Set_the_first_layer(double[] x) {
        for (int i = 0; i < layers[0].length; i++) {
            layers[0][i] = x[i];
            output_of_layers[0][i] = x[i];
        }
    }

    //функция активации - сигмоида
    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    //производная сигмоиды
    private double sigmoid_derivate(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    //прямое распротранение
    public void feed_forward() {
        double buffer = 0;

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < output_of_layers[i + 1].length; j++) {
                for (int z = 0; z < output_of_layers[i].length; z++)
                    buffer += weights[i][j][z] * output_of_layers[i][z];
                    layers[i + 1][j] = buffer + biases[i + 1][j];
                    output_of_layers[i + 1][j] = sigmoid(layers[i+1][j]);
                buffer=0;
            }
        }
    }


    //обратное распространение ошибки
    public void back_forward(double[] y) {
        for (int i = 0; i < output_of_layers[kol_of_layers - 1].length; i++) {
            errors[kol_of_layers - 1][i] = (-y[i] + output_of_layers[kol_of_layers - 1][i]) * sigmoid_derivate(layers[kol_of_layers - 1][i]);
        }

        double buffer = 0;

        for (int i = kol_of_layers - 2; i >0; i--)
            for (int k = 0; k < output_of_layers[i].length; k++) {
                for (int z = 0; z < output_of_layers[i + 1].length; z++)
                    buffer += errors[i + 1][z] * weights[i][z][k];
                errors[i][k] = buffer * sigmoid_derivate(layers[i][k]);
                buffer=0;
            }

        for (int i=0;i<kol_of_layers-1;i++)
            for (int k=0;k<output_of_layers[i+1].length;k++)
                for (int z=0;z<output_of_layers[i].length;z++) {
                    gradient_of_weights[i][k][z]+=errors[i+1][k]*output_of_layers[i][z];
                }

        for (int i=1;i<kol_of_layers;i++)
            for (int j=0;j<biases[i].length;j++)
                gradient_of_biases[i][j]+=errors[i][j];
    }


    //изменение параметров сети c применением момента
    public void change_weights() {
        for (int i = 0; i < kol_of_layers - 1; i++)
            for (int k = 0; k < output_of_layers[i + 1].length; k++)
                for (int z = 0; z < output_of_layers[i].length; z++) {
                    weights[i][k][z] += -gradient_of_weights[i][k][z] * learn_rate / (double) mini_batch + Lambda * (weights[i][k][z] - previous_weights[i][k][z]);
                    gradient_of_weights[i][k][z] = 0;
                    previous_weights[i][k][z] = weights[i][k][z];
                }

        for (int i = 1; i < kol_of_layers - 1; i++)
            for (int j = 0; j < biases[i].length; j++) {
                biases[i][j] -= gradient_of_biases[i][j] * learn_rate / (double) mini_batch;
                gradient_of_biases[i][j] = 0;
            }
    }

    //ошибка тренировочной выборки
    public double error_prognosation_train(double train[]){
        int ind=0;
        double estimates[]=new double[train.length-layers[0].length];
        double x[]=new double[layers[0].length];
        while(ind+layers[0].length!=train.length){
            for (int i=0;i<layers[0].length;i++)
                x[i]=train[i+ind];
            Set_the_first_layer(x);
            feed_forward();
            estimates[ind]=output_of_layers[kol_of_layers-1][0];
            ind++;
        }
        double new_train[]=new double[train.length-layers[0].length];
        for (int i=0;i<train.length-layers[0].length;i++)
            new_train[i]=train[i+layers[0].length];
        return error_prognosation(estimates,new_train);
    }
    //ошибка прогнозирования
    public double error_prognosation(double estimates[],double real_data[]){

        double error=0;

       /* for (int i=0;i<estimates.length;i++)
            error+=Math.abs(real_data[i]-estimates[i])*100/real_data[i];
        error/=estimates.length;*/
        for (int i = 0; i < real_data.length; i++) {
               error+=Math.pow(estimates[i]-real_data[i],2);
        }

        error=Math.sqrt(error);

        error/=estimates.length;
        return error;

    }

    //получений оценок для тестовой выборки
    public double[] get_estimates(double data_train[],double data_test[]){
        int ind=0;
        double estimates[]=new double[layers[0].length+data_test.length];
        double x[]=new double [layers[0].length];
        for (int i=0;i<layers[0].length;i++)
            estimates[i]=data_train[data_train.length-layers[0].length+i];
        for (int i=0;i<data_test.length;i++)
            estimates[i+layers[0].length]=data_test[i];

        double estimates_test[]=new double[data_test.length];

        ind=0;

        while(ind!=data_test.length){
            for (int j=0;j<layers[0].length;j++)
                x[j]=estimates[j+ind];
            Set_the_first_layer(x);
            feed_forward();
           // estimates[ind+layers[0].length]=output_of_layers[kol_of_layers-1][0];
            estimates_test[ind]=output_of_layers[kol_of_layers-1][0];
            ind++;
        }
       /*for (int i=0;i<data_test.length;i++) {
            estimates_test[i] = estimates[i + layers[0].length];
        }*/
        error_prognosation(estimates_test,data_test);
        return estimates_test;
    }

    //обучения для прогнозирования
    public void prognos_train(double data_train[],double data_test[]) {
        System.out.println("Подождите, идет процесс обучения сети!");
        double y[] = new double[layers[kol_of_layers - 1].length];
        double x[] = new double[layers[0].length];
        double estimates[] = new double[data_test.length];
        int ind = 0;
        int m=0;
        try {
            FileWriter fileWriter = new FileWriter("Запись ошибок.txt", false);
            for (int k = 0; k < number_of_epoch; k++) {
                while (ind + layers[0].length != data_train.length) {
                    for (int j = 0; j < layers[0].length; j++)
                        x[j] = data_train[j + ind];
                    if (ind + layers[0].length == data_train.length)
                        for (int j = 0; j < y.length; j++)
                            y[j] = data_test[j];
                    else
                        for (int j = 0; j < y.length; j++)
                            y[j] = data_train[ind + layers[0].length];

                    Set_the_first_layer(x);
                    feed_forward();
                    back_forward(y);
                    m++;
                    if (m == mini_batch) {
                        m = 0;
                        change_weights();
                    }
                    if (k%100==0) {
                        fileWriter.write(String.valueOf(error_prognosation_train(data_train)));
                        fileWriter.write("\n");
                    }
                    //  estimates = get_estimates(data_test, data_real_data);
                    ind++;
                }
                ind = 0;
            }
        }catch (IOException e) {
            e.printStackTrace();}

    }
}
