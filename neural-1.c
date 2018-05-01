#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>




double RandomNumber(float Min, float Max)
{
    return (((float)rand() / (float)RAND_MAX) * (Max - Min)) + Min;
}
double sigmoid_activation(double t){

	double result,exp_value;
	exp_value = exp(-t) ;
	result = 1.0/(1.0+exp_value);
	return result;
	
}
double norm_calculation(double a[][20],int row,int col){
	double sum=0.0,value;
	int i,j;
	for(i=0;i<row;i++){
		for(j=0;j<col;j++){
			sum = sum + (a[i][j]*a[i][j]);
		}
	}
     value =  sqrt(sum);
     //printf("%lf ",sum);
     return value;
}

void hypothesis(int train_vector[],double out_inp[],double Wji[][20],double Wkj[][20],double z[],int n){
    int i,j,k;
    float sum;
    out_inp[0]=1.0;
    for (i=0;i<n;i++){
    	sum=0.0;
    	for(j=0;j<17;j++){
            sum = sum + (Wji[j][i] * train_vector[j]);
                 
    	}
 
    	out_inp[i+1] = sigmoid_activation(sum);
    }
    
    for(k=0;k<10;k++){
    	sum=0.0;
    	for(j=0;j<n+1;j++){
        	sum=sum+(Wkj[j][k] * out_inp[j]);
    	}
    	z[k]= sigmoid_activation(sum);
    }
    
   
}
void back_propagation(int target[],int train_vector[],double out_inp[],double Wji[][20],double Wkj[][20],double z[],int n,float eta){

	double delta_output[20];
	double delta_hidden[20];
	double delta_Wji[20][20];
	double delta_Wkj[20][20];
    int i,j,k;
    for(k=0; k<10; k++){
    	delta_output[k] =  ( target[k]-z[k] ) * ( z[k]*(1.0-z[k]) );
    }
	for(j=0; j<n+1; j++){
	   for(k=0; k<10; k++){
		  	delta_Wkj[j][k] = eta * out_inp[j] * delta_output[k];
		}
	}

		for(j=1; j<n+1; j++){
			double sum = 0.0;
			for(k=0; k<10; k++){
				sum = sum+( delta_output[k] * Wkj[j][k] * ( out_inp[j]*(1.0-out_inp[j]) ) );
			}
			delta_hidden[j-1] = sum;
		}
		//update of Delta weight between input and hidden layer  /_\Wji
		for(i=0; i<17; i++){
			for( j=0; j<n; j++){
				delta_Wji[i][j] = eta*train_vector[i]*delta_hidden[j];
			}
		}
		//if(key==1)
			for(i=0;i<17;i++){
				for(j=0;j<n;j++){
					Wji[i][j] = Wji[i][j] + delta_Wji[i][j];  
				}
			}

			for(i=0;i<n+1;i++){
				for(j=0;j<10;j++){
					Wkj[i][j] = Wkj[i][j] + delta_Wkj[i][j];
				}
			}
	
}
void norm_back_propagation(int target[],int train_vector[],double out_inp[],double dup_Wji[][20],double dup_Wkj[][20],double Wji[][20],double Wkj[][20],double z[],int n,float eta){

	double delta_output[20];
	double delta_hidden[20];
	double delta_Wji[20][20];
	double delta_Wkj[20][20];
    int i,j,k;
    for(k=0; k<10; k++){
    	delta_output[k] =  ( target[k]-z[k])* ( z[k]*(1.0-z[k])  );
    }
	for(j=0; j<n+1; j++){
	   for(k=0; k<10; k++){
		  	delta_Wkj[j][k] = eta * out_inp[j] * delta_output[k];
		}
	}

		for(j=1; j<n+1; j++){
			double sum = 0.0;
			for(k=0; k<10; k++){
				sum = sum+( delta_output[k] * Wkj[j][k] * ( out_inp[j]*(1.0-out_inp[j]) ) );
			}
			delta_hidden[j-1] = sum;
		}
		//update of Delta weight between input and hidden layer  /_\Wji
		for(i=0; i<17; i++){
			for( j=0; j<n; j++){
				delta_Wji[i][j] = eta*train_vector[i]*delta_hidden[j];
			}
		}

	for(i=0;i<17;i++){
		for(j=0;j<n;j++){
			dup_Wji[i][j] = dup_Wji[i][j] + delta_Wji[i][j];  
		}
	}

	for(i=0;i<n+1;i++){
		for(j=0;j<10;j++){
			dup_Wkj[i][j] = dup_Wkj[i][j] + delta_Wkj[i][j];
		}
	}
	
}
void getfiledata(int array[][40],int class[],int len,char str[]){

		int i,j;
		FILE *f = fopen(str, "r");
		while(getc(f)!= EOF){                                       // Taking training set from file  
			for( i=0;i<len;i++){
				for( j=0;j<17;j++){
					if(j==0){
						fscanf(f,"%d",&class[i]);
						array[i][j] = 1;
					}
					else{
						fscanf(f,"%d",&array[i][j]);
				    }
				}
			}
		}
    fclose(f);  
    
}

int main(){

	double z[20];
	double Wji[20][20];
	double Wkj[20][20];
	double dup_Wji[20][20];
	double dup_Wkj[20][20];
	double out_inp[20];
	int test_vector[3000][40]; 
	int train_vector[3000][40];
	int train_class[3000];
	int test_class[3000];
	int label_comp[3000];
 	int i,j,len=2216,count=0,test_len=998;



 	getfiledata(train_vector,train_class,len,"train1.txt");  
    

    int n,flag,epoch=0,target[20];
    printf("Enter number of neurons in hidden layer:");
    scanf("%d",&n); 

    printf("Select 1)Epoch\n 2)Norm\n 3)Entropy");
    scanf("%d",&flag);

    for(i=0;i<17;i++){
    	for(j=0;j<n;j++){
    		Wji[i][j]= RandomNumber(-0.01,0.01);
    	}
    }
    for(i=0;i<n+1;i++){
    	for(j=0;j<10;j++){
    		Wkj[i][j]= RandomNumber(0.01,0.05);
    	}
    }
if(flag==1){
    for(epoch=0;epoch<5000;epoch++){
    	for(i=0;i<len;i++){
    		for(j=0;j<10;j++){
    	      target[j]=0;
            }
    		hypothesis(train_vector[i],out_inp,Wji,Wkj,z,n);
    		target[train_class[i]-1]=1;
    		back_propagation(target,train_vector[i],out_inp,Wji,Wkj,z,n,0.001);
    	}  
   }
   
}
	else if(flag==2){
		double epsila=99999.0,x1,x2;
 	    int epho_count=0;
		while(epsila>0.01 ){
			for(i=0;i<20;i++){
 				for(j=0;j<20;j++){
 					dup_Wkj[i][j]=0;
 					dup_Wji[i][j]=0; 			
 				}
 			}					
			for(i=0;i<len;i++){
				for(j=0;j<10;j++){
    	      		target[j]=0;
            	}
				hypothesis(train_vector[i],out_inp,Wji,Wkj,z,n);
    			target[train_class[i]-1]=1;
    			norm_back_propagation(target,train_vector[i],out_inp,dup_Wji,dup_Wkj,Wji,Wkj,z,n,0.001);				
			}
			for(i=0;i<17;i++){
				for(j=0;j<n;j++){
					Wji[i][j] = Wji[i][j] + dup_Wji[i][j];  
				}
			}	

			for(i=0;i<n+1;i++){
				for(j=0;j<10;j++){
					Wkj[i][j] = Wkj[i][j] + dup_Wkj[i][j];
				}	
			}
			x1 = norm_calculation(dup_Wkj,n+1,10);
			x2 = norm_calculation(dup_Wji,17,n);
			 
			epsila = (x1+x2)/2.0;
			epho_count++;	
			printf("%d %lf\n",epho_count,epsila);


		}
	}

    getfiledata(test_vector,test_class,test_len,"test.txt");

    for(i=0;i<test_len;i++){
    	float max=0.0;
    	int index;
    	hypothesis(test_vector[i],out_inp,Wji,Wkj,z,n);
    	for(j=0;j<10;j++){
    		if(max<z[j]){
    			max=z[j];
    			index=j;
    		}	
    		printf("%.7lf ",z[j]);
    	}
        label_comp[i]=index+1;      
    	printf("%d\n",index+1);
    }
    for(i=0;i<test_len;i++){
    	if(test_class[i]==label_comp[i]){
    		count++;
    	}
    }
    printf("Accuracy %lf\n",(count/998.0)*100.0);

return 0;
}
