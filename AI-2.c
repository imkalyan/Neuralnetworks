#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <limits.h>
double dwji[17][5],dwkj[6][10];

double sigmoid(double no){
    return (1/(1+exp(-no)));
}
double norm_calculation(int nrow, int ncol,double A[][ncol]){
    int i,j;
    double norm;
    double sum=0;
    for(i=0; i<nrow; i++){
        for(j=0; j<ncol; j++){
            sum = sum+ (A[i][j]*A[i][j]);
        }
    }
    norm=sqrt((double)sum);
    return (norm);    
}

//z,i_h,h_o,tset,x,y
void forwardpropagation(double z[10],double i_h[17][5],double h_o[6][10],int tset[17],double x_f[18],double y_h[6]){
    x_f[0]=1.0;
    y_h[0]=1.0;
    double temp_z=0,temp_y=0;
    for(int m=1;m<17;m++){
        x_f[m] = tset[m]*1.0 ;
    }
    
    for(int i=0;i<10;i++){
        temp_z=0;
        for(int m=0;m<6;m++){
            temp_y=0;
            if(m!=5){
                for(int n=0;n<17;n++){
                    temp_y  =   i_h[n][m]   *  x_f[n] + temp_y;
                }
                temp_y=sigmoid(temp_y);
                y_h[m+1]=temp_y;
            }
            temp_z = h_o[m][i]  *  y_h[m] + temp_z;
        }
        temp_z=sigmoid(temp_z);
        z[i]=temp_z;
    }
}

//tset,y,x,z,h_o,dwji,dwkj
void backwardpropagation(int tset[17],double y_h[6],double x_f[18],double z[10],double h_o[6][10],double dwji[17][5],double dwkj[6][10]){
    double diff_z[10],diff_y[6],deviation[10],t[10]={0},delta_h[10],delta_f[10],temp=0;
    int place;
    place = tset[0];
    place = place-1;
    t[place] = 1;
    
    for(int i=0;i<10;i++){
        diff_z[i] = z[i]*(1-z[i]);
        deviation[i] = t[i]-z[i];
        delta_h[i] = deviation[i] * diff_z[i];
        if(i<6){
            diff_y[i] = y_h[i]   *   (1-y_h[i]);
        }
    }
    
    for(int j=0;j<6;j++){
        for(int k=0;k<10;k++){
            dwkj[j][k] = 0.01 * deviation[k]  *  diff_z[k]  *  y_h[j];
        }
    }
    
    for(int j=1;j<6;j++){
        temp=0.0;
        for(int r=0;r<10;r++){
            temp  = delta_h[r]  *  h_o[j][r]  *diff_y[j]  +  temp;
        }
        delta_f[j-1] = temp;
    }
    for(int i=0;i<17;i++){
        for(int j=0;j<5;j++){
            dwji[i][j] = 0.01 * delta_f[j]  *  x_f[i];
        }
    }
}

int main()
{  
    char buffer[1024] ;   
    char *record,*line;   
    int i=0,j=0,tset[17],h_s = 5,k=0,p=0;
    double z[10],i_h[17][5],h_o[6][10],x[18],y[6];     
    srand(time(NULL));
    int r_x,r_y;
    r_x=-10;
    r_y=+10;
    for(int m=0;m<17;m++){
        for(int n=0;n<5;n++){
            i_h[m][n]=((rand()%(r_y-r_x+1))+r_x)/1000.0;

        }
    }

    for(int m=0;m<6;m++){
        for(int n=0;n<10;n++){
            h_o[m][n]=((rand()%(r_y-r_x+1))+r_x)/1000.0;
        }
    }

    

        j=0; 
        int t=0;
        FILE *f = fopen("train1.txt", "r");
        int train_vector[3000][40],len=2216;

        while(getc(f)!= EOF){                                       // Taking training set from file  
            for( i=0;i<len;i++){
                for( j=0;j<17;j++){
                        fscanf(f,"%d",&train_vector[i][j]);
                }
            }
        }
    fclose(f);
    double norm_ofWji;
    //norm_ofWji = norm_calculation(17,5,D_Wji_ofNew);
  
    for(int q=0;q<1000;q++){
        for(int k=0;k<len;k++){
                forwardpropagation(z,i_h,h_o,train_vector[k],x,y);
                backwardpropagation(train_vector[k],y,x,z,h_o,dwji,dwkj);
                for(int m=0;m<17;m++){
                    for(int n=0;n<5;n++){
                        i_h[m][n] = i_h[m][n] + dwji[m][n];
                    }
                }
                for(int m=0;m<6;m++){
                    for(int n=0;n<10;n++){
                        h_o[m][n] = h_o[m][n] + dwkj[m][n];
                    }
                }
        }
    }  

    

    FILE *f1= fopen("test.txt", "r"); 
    int Atest[3000][50]; 
    while(getc(f1)!= EOF){ 
        for(int i=0;i<999;i++){ 
            for(int j=0;j<17;j++){ 
                fscanf(f1,"%d",&Atest[i][j]); 
            } 
        } 
    }
    int label[999],label_comp[999];
    for(int i=0;i<998;i++){ 
        int Xtest[20];
        Xtest[0]=1; 
             for(int j=1; j<17; j++){ 
                 Xtest[j] = Atest[i][j]; 
             } 
             label_comp[i] = Atest[i][0];
        forwardpropagation(z,i_h,h_o,Xtest,x,y);
        float max=INT_MIN * 1.0;
        int index; 
         
        for(int p=0;p<10;p++){ 
           printf("%lf  ",z[p]);
            
                if(z[p] > max && z[p]<0.5){
                    max = z[p]; 
                    index = p; 
                } 
        } 

        printf("index: %d\n",index + 1); 
        label[i]=index+1; 
        }
        int count=0;
    for(int a=0;a<998;a++){
        if(label[a]==label_comp[a]){
            count++;
        }
    }
    printf("The accuracy %f\n",(count/998.0)*100.0);    
	return 0 ; 
}