#include <iostream>
#include <vector>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/Jacobi>

using Eigen::MatrixXd;
using Eigen::MatrixXf;


int main (){
 
    MatrixXd x1(1, 3),
             x2(1, 3),
             x3(1, 3),
             x4(1, 3),
             x9(1, 3),
             x10(1, 3),
             x11(1, 3),
             x12(1, 3);
    
    MatrixXd y1(1, 3),
             y2(1, 3),
             y3(1, 3),
             y4(1, 3),
             y9(1, 3),
             y10(1, 3),
             y11(1, 3),
             y12(1, 3);
  
    // osam tacaka za fundamentalnu matricu F = FF

    x1  << 958, 38, 1;     
    y1  << 933, 33, 1;    
    x2  << 1117, 111, 1;
    y2  << 1027, 132, 1;
    x3  << 874, 285, 1;
    y3  << 692, 223, 1;
    x4  << 707, 218, 1;
    y4  << 595, 123, 1;
    x9  << 292, 569, 1;
    y9  << 272, 360, 1;    
    x10 << 770, 969, 1;
    y10 << 432, 814, 1;
    x11 << 770, 1465, 1;
    y11 << 414, 1284, 1;    
    x12 << 317, 1057, 1;    
    y12 << 258, 818, 1;
    

    MatrixXd xx(8, 3), yy(8, 3);
       
    xx<< x1, x2, x3, x4, x9, x10, x11, x12 ;
    yy<< y1, y2, y3, y4, y9, y10, y11, y12 ;
    
    // std::cout << xx << std::endl;
    // std::cout << yy << std::endl;
    
    // y^T F x = 0
    // JED [{a1, a2, a3}, {b1, b2, b3}]= { a1b1, a2b1, a3b1, a1b2, a2b2, a3b2, a1b3,a2b3,a3b3};
        
    MatrixXd jed8 (8, 9);
    for ( int i = 0 ; i < 8 ; i++){
       int  k = 0, l =0;
        for (int j = 0 ; j < 9 ; j++){
            
            jed8(i, j) = xx(i, k) * yy (i, l); 
            k++ ;
            if(k == 3){
                
                k=0;
                l++;
            }   
        }
    }
   
//     std::cout<< jed8<<std::endl;
    
    //racunamo SVD: 

    Eigen::JacobiSVD<MatrixXd> SVDjed8(jed8, Eigen::ComputeFullU|Eigen::ComputeFullV);
    MatrixXd SVDjed8V = SVDjed8.matrixV();        
    //     std::cout<<SVDjed8V<<"\n";          
    //uzimamo samo poslednje vrednosti, to nas jedino interesuje: 
    MatrixXd p(1, 9);
    
    for(int i = 0 ; i < 9 ; i++){
        p(i) = SVDjed8V(i, 8);   
    }
            
    //std::cout<< p << std::endl;
   
   // pravimo matricu ff :
    MatrixXd ff(3, 3);
    ff << p(0), p(1), p(2),
          p(3), p(4), p(5),
          p(6), p(7), p(8);
    
    //std::cout<< ff <<std::endl;
    
    //std::cout<< ff.determinant();
    
    Eigen::JacobiSVD<MatrixXd> SVDFF(ff, Eigen::ComputeFullU | Eigen::ComputeFullV);

    MatrixXd SVDffV = SVDFF.matrixV();        
    MatrixXd SVDffU = SVDFF.matrixU();       
    MatrixXd SVDffDD = SVDFF.singularValues();        
/*
    for(int i = 0; i < 3; i++){
        SVDffV(1, i) *= (-1); 
        SVDffU(1, i) *= (-1); 
    }*/
    // treca kolona je trazeni epipol e1:
    MatrixXd e1(1, 3);
    for( int i = 0 ; i < 3 ;i++){
        e1(i) = SVDffV(i, 2);
    }
    // prebacimo epol e1 u afine koordinate:
    for(int i = 0 ; i<3; i++){
        e1(i) = e1 (i) / e1(2);
    }
    
    //std::cout << e1 << std::endl;
 
    // treba nam epipol2- e2 je treca kolona matrice U iz SVD ;
    MatrixXd e2(1, 3);
    for( int i = 0 ; i < 3 ;i++){
        e2(i) = SVDffU(i, 2);
    }
    // prebacimo epipol e2 u afine koordinate:
    for(int i = 0 ; i < 3; i++){
        e2(i) = e2 (i) / e2(2);
    }
    
    //std::cout << e2 << std::endl;   
    
    //pravimo dijagonalnu matricu
    
    MatrixXd dd1 = SVDffDD.array().matrix().asDiagonal();
    dd1(2,2)= 0;
    
    //std::cout<<"dd1: \n"<< dd1<<std::endl;
    MatrixXd ff1 =  SVDffU*dd1*(SVDffV.transpose());
    
    //std::cout<<"\nff:  \n"<<ff << std::endl;
    //std::cout<<"\nff1: \n"<<ff1 <<std::endl;
    
    //std::cout<<ff1.determinant()<<std::endl;
    
    
    
    
    
    return 0;
}
