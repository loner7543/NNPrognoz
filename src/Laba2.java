
import java.io.*;
import java.util.*;

public class Laba2 {
    static double max(double data[][], int i) {
        double max = data[0][0];

        for (int j = 0; j < 130; j++)
            if (data[j][i] > max) max = data[j][i];

        return max;
    }


    public static final int DATA_TEST_LENGTH=50;
    public static void main(String[] args) {
        double buffer;
        FileWriter fileWriter,fileWriter_test;
        String string, s[];
        Perceptrone_model perceptrone_model=new Perceptrone_model(3,5,5,1);


        double data_train[]=new double [200];
        double data_test[]=new double [DATA_TEST_LENGTH];
        double data_all[][]=new double[250][2];


        int l=0;
        try {
            Scanner scanner = new Scanner(new File("Данные для прогноза.txt"));
            Scanner Scanner_train = new Scanner(new File("Data for neural network.txt"));
            l=0;
            int k=0;

            while (scanner.hasNextLine()) {
                string = scanner.nextLine();
                s = string.split("\\,");
                for (int i = 0; i < s.length; i++) {
                    if ((i==2)||(i==3)) {
                        buffer = Double.parseDouble(s[i]);
                        data_all[k][l] = buffer;
                        l++;
                    }
                }
                k++;
                l=0;
            }

            double dif_data[]=new double[250];
            for (int i=0;i<k;i++)
                dif_data[i]=data_all[i][0]-data_all[i][1];

            double square_sums=0;
            for (int i=0;i<k;i++)
                square_sums+=dif_data[i]*dif_data[i];
            square_sums=Math.sqrt(square_sums);

            for (int i=0;i<k;i++)
                dif_data[i]/=square_sums;

            for (int i=0;i<200;i++)
                data_train[i]=dif_data[i];

            for (int i=0;i<DATA_TEST_LENGTH;i++)
                data_test[i]=dif_data[200+i];


            perceptrone_model.prognos_train(data_train,data_test);
            double estimates[];
            estimates=perceptrone_model.get_estimates(data_train,data_test);

            for (int i=0;i<estimates.length;i++)
                System.out.println("Прогнозируемые данные: "+ estimates[i]*square_sums + " Реальные данные: "+data_test[i]*square_sums);
            Scanner_train.close();


        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
