#include <iostream>
#include <vector>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/Jacobi>
#include <GL/glut.h>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::MatrixXf;


static int window_width, window_height, i = 0 ;

static void on_keyboard(unsigned char key, int x, int y);
static void on_reshape(int width, int height);
static void  on_display(void);

MatrixXd rekonstruisane_400(24, 3);
        
MatrixXd izracunavanje();
MatrixXd uAfine(MatrixXd &xx);
MatrixXd triD(MatrixXd &xx, MatrixXd &yy, MatrixXd &t1, MatrixXd &t2);

Eigen::Vector3d nevidljive_proizvod(Eigen::Vector3d & a, Eigen::Vector3d & b,       Eigen::Vector3d & c, Eigen::Vector3d & d, Eigen::Vector3d & e, Eigen::Vector3d & f, Eigen::Vector3d & g, Eigen::Vector3d & h, Eigen::Vector3d & i, Eigen::Vector3d & j);

void nacrtaj_malu( MatrixXd &rekonstruisane_400);
void nacrtaj_srednju(MatrixXd &rekonstruisane_400);
void nacrtaj_veliku( MatrixXd &rekonstruisane_400);
    
void nacrtaj_ose();
int main (int argc, char * argv[]){
    
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 700);
    glutCreateWindow("3D - rekonstrukcija: ");
	glClearColor(1, 1, 1, 1);
    
    glutKeyboardFunc(on_keyboard);
    glutReshapeFunc(on_reshape);
    glutDisplayFunc(on_display);

      
    glutMainLoop();
 
    

    return 0;
}


static void on_keyboard(unsigned char key, int x, int y){
    switch (key) {
        case 27:
            /* Zavrsava se program. */
            exit(0);
            break;
    }
}

static void on_reshape(int width, int height) {
  
    /* Pamte se sirina i visina prozora. */
    window_width = width;
    window_height = height;
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective( 40,
                    window_width/(float)window_height, 1, 500);

    
}

static void on_display(void) {
     /* Brise se prethodni sadrzaj prozora. */
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    
    /* Podesava se tacka pogleda. */
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

   
    gluLookAt( 15, 45, 25,
                0,  0,  0,
                0,  -1,  0
             );

    //nacrtaj_ose();
    
    if( i == 0 ){
        rekonstruisane_400 << izracunavanje()*0.5;
        std::cout << "Rekonstruisane400 : \n" << rekonstruisane_400 << std::endl; 
        i++;
    }
    glPushMatrix();
        nacrtaj_malu(rekonstruisane_400);
        nacrtaj_srednju(rekonstruisane_400);
        nacrtaj_veliku(rekonstruisane_400);
    glPopMatrix();

    glutSwapBuffers();
}



MatrixXd izracunavanje() {

   
    MatrixXd x1(1, 3), x2(1, 3), x3(1, 3), x4(1, 3), x19(1, 3), x20(1, 3), x23(1, 3), x24(1, 3);
    MatrixXd y1(1, 3), y2(1, 3), y3(1, 3), y4(1, 3), y19(1, 3), y20(1, 3), y23(1, 3), y24(1, 3);
  
    // osam tacaka za fundamentalnu matricu F = FF

    
     x1 << 331,  75, 1;     
     y1 << 389,  76, 1;    
     x2 << 495,  53, 1;
     y2 << 561,  75, 1;
     x3 << 716, 166, 1;
     y3 << 566, 199, 1;
     x4 << 538, 191, 1;
     y4 << 371, 195, 1;
    x19 << 924, 600, 1;    
    y19 << 861, 655, 1;
    x20 << 700, 779, 1;    
    y20 << 456, 778, 1;
    x23 << 918, 787, 1;    
    y23 << 857, 838, 1;
    x24 << 697, 988, 1;    
    y24 << 461, 977, 1;
    

    MatrixXd xx(8, 3), yy(8, 3);
       
    xx<< x1, x2, x3, x4, x19, x20, x23, x24 ;
    yy<< y1, y2, y3, y4, y19, y20, y23, y24 ;
    
    // std::cout << xx << std::endl;
    // std::cout << yy << std::endl;
    
    // y^T F x = 0
    // JED [{a1, a2, a3}, {b1, b2, b3}]= { a1b1, a2b1, a3b1, a1b2, a2b2, a3b2, a1b3,a2b3,a3b3};
        
    MatrixXd jed8 (8, 9);
    for ( int i = 0 ; i < 8 ; i++) {
       int  k = 0, l =0;
        for (int j = 0 ; j < 9 ; j++) {

            jed8(i, j) = xx(i, k) * yy (i, l); 
            k++ ;
            if(k == 3){
                
                k=0;
                l++;
            }   
        }
    }
   
    // std::cout<< jed8<<std::endl;
    
    //racunamo SVD: 

    Eigen::JacobiSVD<MatrixXd> SVDjed8(jed8, Eigen::ComputeFullU | Eigen::ComputeFullV);
    MatrixXd SVDjed8V = SVDjed8.matrixV();        
    
    //uzimamo samo poslednje vrednosti, to nas jedino interesuje: 
    MatrixXd p(1, 9);
    
    for(int i = 0 ; i < 9 ; i++){
        p(i) = SVDjed8V(i, 8);   
    }
            
    //std::cout<< p << std::endl;
   
   // pravimo matricu ff 3x3 :
    MatrixXd ff(3, 3);
    ff << p(0), p(1), p(2),
          p(3), p(4), p(5),
          p(6), p(7), p(8);
    
    //std::cout<< ff <<std::endl;
    
    //std::cout<< ff.determinant();
    
    Eigen::JacobiSVD<MatrixXd> SVDFF(ff, Eigen::ComputeFullU | Eigen::ComputeFullV);

    MatrixXd SVDffV  = SVDFF.matrixV();        
    MatrixXd SVDffU  = SVDFF.matrixU();       
    MatrixXd SVDffDD = SVDFF.singularValues();        
    
    // treca kolona je trazeni epipol e1:
    MatrixXd e1(1, 3);
    for( int i = 0 ; i < 3 ; i++) {
        e1(i) = SVDffV(i, 2);
    }
    // prebacimo epol e1 u afine koordinate:
    for(int i = 0 ; i < 3 ; i++) {
        e1(i) = e1 (i) / e1(2);
    }
    
    //std::cout << e1 << std::endl;
 
    // treba nam epipol2 -> e2 je treca kolona matrice U iz SVD ;
    MatrixXd e2(1, 3);
    for( int i = 0 ; i < 3 ;i++) {
        e2(i) = SVDffU(i, 2);
    }
    // prebacimo epol e2 u afine koordinate:
    for(int i = 0 ; i < 3 ; i++) {
        e2(i) = e2 (i) / e2(2);
    }
    
    //std::cout << e2 << std::endl;   
    
    //pravimo dijagonalnu matricu
    MatrixXd dd1 = SVDffDD.array().matrix().asDiagonal();
    dd1(2,2) = 0;
    
    //std::cout<<"dd1: \n"<< dd1<<std::endl;
    MatrixXd ff1 =  SVDffU*dd1*(SVDffV.transpose());
    
    //std::cout<<"\nff:  \n"<<ff << std::endl;
    //std::cout<<"\nff1: \n"<<ff1 <<std::endl;
    
    //std::cout<<ff1.determinant()<<std::endl;
    
    
    /* REKONSTRUKCIJA SKRIVENIH TACAKA:  */    
    Eigen::Vector3d x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_21, x_22;
    Eigen::Vector3d y_5, y_6, y_7, y_8, y_9, y_10, y_11, y_12, y_13, y_14, y_15, y_16, y_17, y_18, y_21, y_22;
    
    // potrebne su nam i prethodne tacke kao vektori, zbog vektorskog mnozenja! 

    Eigen::Vector3d x_1, x_2, x_3, x_4, x_19, x_20, x_23, x_24;
    Eigen::Vector3d y_1, y_2, y_3, y_4, y_19, y_20, y_23, y_24;
    
    
    x_1  <<  x1(0),  x1(1),  x1(2);
    y_1  <<  y1(0),  y1(1),  y1(2);   
    x_2  <<  x2(0),  x2(1),  x2(2);
    y_2  <<  y2(0),  y2(1),  y2(2);       
    x_3  <<  x3(0),  x3(1),  x3(2);
    y_3  <<  y3(0),  y3(1),  y3(2);   
    x_4  <<  x4(0),  x4(1),  x4(2);
    y_4  <<  y4(0),  y4(1),  y4(2);       
    x_19 << x19(0), x19(1), x19(2);
    y_19 << y19(0), y19(1), y19(2);  
    x_20 << x20(0), x20(1), x20(2);
    y_20 << y20(0), y20(1), y20(2);  
    x_23 << x23(0), x23(1), x23(2);
    y_23 << y23(0), y23(1), y23(2);    
    x_24 << x24(0), x24(1), x24(2);
    y_24 << y24(0), y24(1), y24(2);
    
    x_5  << 330, 294, 1;
    x_7  << 714, 400, 1;
    y_7  << 567, 424, 1;
    x_8  << 553, 430, 1;
    y_8  << 378, 421, 1;   
    x_9  << 262, 339, 1;
    y_9  << 281, 311, 1;    
    y_10 << 713, 330, 1;
    x_11 << 774, 369, 1;
    y_11 << 688, 406, 1;
    x_12 << 312, 412, 1;
    y_12 << 234, 378, 1;
    x_13 << 262, 586, 1;
    y_14 << 718, 569, 1;
    x_15 << 770, 618, 1;
    y_15 << 686, 642, 1;
    x_16 << 312, 666, 1;    
    y_16 << 247, 615, 1;
    x_17 <<  91, 631, 1;    
    y_17 << 122, 552, 1;
    x_21 <<  95, 825, 1;    
    y_21 << 128, 721, 1;
 
    y_5 = nevidljive_proizvod(y_8, y_4, y_7, y_3, y_1,
                              y_4, y_1, y_3, y_2, y_8 );
    
    x_6 = nevidljive_proizvod( x_5, x_1, x_8, x_4, x_2,
                               x_8, x_5, x_3, x_2, x_7 );
    
    y_6 = nevidljive_proizvod( y_5, y_1, y_8, y_4, y_2, 
                               y_8, y_5, y_3, y_2, y_7 );
        

    x_10 = nevidljive_proizvod( x_16, x_13, x_12,  x_9, x_11,
                                x_12, x_11, x_16, x_15,  x_9 );
    
    y_13 = nevidljive_proizvod( y_15, y_16, y_10,  y_9, y_14,
                                y_16, y_12, y_15, y_11,  y_9 );
     
    x_14 = nevidljive_proizvod( x_16, x_15, x_12, x_11, x_13,
                                x_16, x_13, x_12,  x_9, x_15 ); 
    
    x_18 = nevidljive_proizvod( x_20, x_19, x_24, x_23, x_17, 
                                x_24, x_21, x_20, x_17, x_19 );
    
    y_18 = nevidljive_proizvod( y_20, y_19, y_24, y_23, y_17, 
                                y_24, y_21, y_20, y_17, y_19 );
    
    
    
    x_22 = nevidljive_proizvod( x_20, x_19, x_24, x_23, x_21,
                                x_24, x_21, x_20, x_17, x_23);
    
    y_22 = nevidljive_proizvod( y_20, y_19, y_24, y_23, y_21,
                                y_24, y_21, y_20, y_17, y_23); 
    
    
    //potreban nam je dalje rad sa matricama: 
    MatrixXd x5(1, 3), x6(1, 3), x7(1, 3), x8(1, 3), x9(1, 3), x10(1, 3), x11(1, 3), x12(1, 3), x13(1, 3), x14(1, 3), x15(1, 3), x16(1, 3), x17(1, 3), x18(1, 3), x21(1, 3), x22(1, 3);
   
    MatrixXd y5(1, 3), y6(1, 3), y7(1, 3), y8(1, 3), y9(1, 3), y10(1, 3), y11(1, 3), y12(1, 3), y13(1, 3), y14(1, 3), y15(1, 3), y16(1, 3), y17(1, 3), y18(1, 3), y21(1, 3), y22(1, 3);
    
     x5 <<  x_5(0),  x_5(1),  x_5(2) ;
     x6 <<  x_6(0),  x_6(1),  x_6(2) ;
     x7 <<  x_7(0),  x_7(1),  x_7(2) ;
     x8 <<  x_8(0),  x_8(1),  x_8(2) ;    
     x9 <<  x_9(0),  x_9(1),  x_9(2) ;
    x10 << x_10(0), x_10(1), x_10(2) ;
    x11 << x_11(0), x_11(1), x_11(2) ;
    x12 << x_12(0), x_12(1), x_12(2) ;
    x13 << x_13(0), x_13(1), x_13(2) ;
    x14 << x_14(0), x_14(1), x_14(2) ;
    x15 << x_15(0), x_15(1), x_15(2) ;
    x16 << x_16(0), x_16(1), x_16(2) ;
    x17 << x_17(0), x_17(1), x_17(2) ;
    x18 << x_18(0), x_18(1), x_18(2) ;
    x21 << x_21(0), x_21(1), x_21(2) ;
    x22 << x_22(0), x_22(1), x_22(2) ;
    
    
    
     y5 <<  y_5(0),  y_5(1),  y_5(2) ; 
     y6 <<  y_6(0),  y_6(1),  y_6(2) ;
     y7 <<  y_7(0),  y_7(1),  y_7(2) ;
     y8 <<  y_8(0),  y_8(1),  y_8(2) ;
     y9 <<  y_9(0),  y_9(1),  y_9(2) ;
    y10 << y_10(0), y_10(1), y_10(2) ;
    y11 << y_11(0), y_11(1), y_11(2) ;
    y12 << y_12(0), y_12(1), y_12(2) ;
    y13 << y_13(0), y_13(1), y_13(2) ;
    y14 << y_14(0), y_14(1), y_14(2) ;
    y15 << y_15(0), y_15(1), y_15(2) ;
    y16 << y_16(0), y_16(1), y_16(2) ;
    y17 << y_17(0), y_17(1), y_17(2) ;
    y18 << y_18(0), y_18(1), y_18(2) ;
    y21 << y_21(0), y_21(1), y_21(2) ;
    y22 << y_22(0), y_22(1), y_22(2) ;
        
    
    
    /* TRIANGULACIJA:  */
    MatrixXd m = MatrixXd::Identity(3, 3);
    MatrixXd t1 (3, 4);
    t1 << m.row(0), 0,
          m.row(1), 0,
          m.row(2), 0;  
          
    MatrixXd E2 (3, 3), E_2(3, 3);
    E2 <<     0, -e2(2),  e2(1),
          e2(2),      0, -e2(0),
         -e2(1),  e2(0),      0;
    
    E_2 = E2 * ff1;

    MatrixXd t2 (3, 4);
    t2 << E_2.row(0), e2(0),
          E_2.row(1), e2(1),
          E_2.row(2), e2(2);  
      
    //std::cout << t2<<"\n";
    
    /* za svaku tacku dobijamo sistem od 4 jednacine sa 4 homogene nepoznate:
    jednacine[x1, y1] = x1(2)*t1(3) -  x1(3)*t1(2), 
                       -x1(1)*t1(3) +  x1(3)*t1(1),
                        y1(2)*t2(3) -  y1(3)*t2(2),
                       -y1(1)*t2(3) +  y1(3)*t2(1);
    
    */
    
    // racunamo svd za svaku tacku :
    MatrixXd slika1(24, 3), slika2(24, 3) ;
    slika1 << x1,  x2,  x3,  x4,  x5,  x6,  x7,  x8,  x9,
             x10, x11, x12, x13, x14, x15, x16, x17, x18,
             x19, x20, x21, x22, x23, x24 ;
    slika2 << y1,  y2,  y3,  y4,  y5,  y6,  y7,  y8,  y9,
             y10, y11, y12, y13, y14, y15, y16, y17, y18,
             y19, y20, y21, y22, y23, y24 ;
   
    MatrixXd jednacine(24, 4), prva_pom(1, 3), druga_pom(1, 3);
    for(int i = 0; i < 24 ; i++){
        prva_pom << slika1.row(i);
        druga_pom << slika2. row(i);
        jednacine.row(i) << triD (prva_pom, druga_pom, t1, t2);
    }
    // prebacujemo koordinate u afine :
    MatrixXd rekonstruisane(24, 3), pom(1, 4);
    for (int i = 0; i < 24; i++){
        pom << jednacine.row(i);
        rekonstruisane.row(i) << uAfine(pom); 
    }
    
    // mnozenje z- koordinate ( nije radjena normalizacija! )
    MatrixXd rekonstruisane400(24, 3);
    for(int i = 0 ; i < 24 ; i++){
        rekonstruisane400.row(i) << rekonstruisane(i, 0), rekonstruisane(i, 1), rekonstruisane(i, 2)*400 ;
    }
 
    return rekonstruisane400;

}

MatrixXd uAfine(MatrixXd &xx) {
    MatrixXd afine(1, 3);
    afine << xx(0) / xx(3), xx(1) / xx(3), xx(2) / xx(3) ;
    
    return afine;
}


MatrixXd triD(MatrixXd &xx, MatrixXd &yy, MatrixXd &t1, MatrixXd &t2) {

     MatrixXd jednacina1(4, 4);
     jednacina1.row(0)<<      xx(1)*t1.row(2)  - xx(2)*t1.row(1);
     jednacina1.row(1)<< (-1)*xx(0)*t1.row(2)  + xx(2)*t1.row(0); 
     jednacina1.row(2)<<      yy(1)*t2.row(2)  - yy(2)*t2.row(1);
     jednacina1.row(3)<< (-1)*yy(0)*t2.row(2)  + yy(2)*t2.row(0);
     //std::cout<<jednacina1<<"\n"; 
     
     Eigen::JacobiSVD<MatrixXd> svdJedn(jednacina1, Eigen::ComputeFullU | Eigen::ComputeFullV);
  
     MatrixXd svdV = svdJedn.matrixV();        
     MatrixXd svdU = svdJedn.matrixU();       
     MatrixXd svdDD = svdJedn.singularValues();        
     //std::cout<<svdV<<"\n";
   
     MatrixXd jedn(1, 4);
     for( int i = 0 ; i < 4 ;i++){
         jedn(i) = svdV(i, 3);
     }
     //std::cout<<jedn<<"\n";    
     return jedn;
}
    
void nacrtaj_ose(){
    
  glBegin(GL_LINES);
    glColor3f(1,0,0);
    glVertex3f(0, 0, 0);
    glVertex3f(17, 0, 0);

  glEnd();
    
  glBegin(GL_LINES);
  
    glColor3f(0,1,0);
    glVertex3f(0,0,0);
    glVertex3f(0,17,0);
    
  glEnd();
    
  glBegin(GL_LINES);
  
    glColor3f(0,0,1);
    glVertex3f(0,0,0);
    glVertex3f(0,0,17);
    
  glEnd();
      
}

/*mala kutija */                 
void nacrtaj_malu( MatrixXd &rekonstruisane_400) {
    
    MatrixXd malaIvice (12, 2);
    malaIvice << 0, 1,
                 1, 2,
                 2, 3,
                 3, 0,
                 4, 5,
                 5, 6,
                 6, 7,
                 7, 4,
                 0, 4,
                 1, 5,
                 2, 6,
                 3, 7 ;
    float x, y;
    for ( int i = 0 ; i < 12 ; i++ ) {
        x = malaIvice(i, 0);
        y = malaIvice(i, 1);
        glBegin(GL_LINES);
            glColor3f(0, 0, 1);
            glVertex3f(rekonstruisane_400(x, 0),rekonstruisane_400(x, 1), rekonstruisane_400(x, 2) );
            glVertex3f(rekonstruisane_400(y, 0),rekonstruisane_400(y, 1), rekonstruisane_400(y, 2));
        glEnd();
        
    }
    
}


void nacrtaj_srednju(MatrixXd &rekonstruisane_400) {
    MatrixXd srednjaIvice (12, 2);
    srednjaIvice <<  8,  9,
                     9, 10,
                    10, 11,
                    11,  8,
                    12, 15,
                    15, 14,
                    14, 13,
                    13, 12,
                    15, 11,
                    14, 10,
                     8, 12,
                    13,  9;
    float x, y;
    for ( int i = 0 ; i < 12 ; i++ ) {
        x = srednjaIvice(i, 0);
        y = srednjaIvice(i, 1);
        glBegin(GL_LINES);
            glColor3f(1, 0, 1);
            glVertex3f(rekonstruisane_400(x, 0),rekonstruisane_400(x, 1), rekonstruisane_400(x, 2) );
            glVertex3f(rekonstruisane_400(y, 0),rekonstruisane_400(y, 1), rekonstruisane_400(y, 2));
        glEnd();
        
    }
}

/*velika kutija */                 
void nacrtaj_veliku( MatrixXd &rekonstruisane_400) {
    
    MatrixXd velikaIvice (12, 2);
    velikaIvice <<  
                    16, 17, 
                    17, 18,
                    18, 19,
                    19, 16,
                    20, 21,
                    21, 22,
                    22, 23,
                    23, 20,
                    16, 20,
                    17, 21,
                    18, 22,
                    19, 23;
    
    float x, y;
    for ( int i = 0 ; i < 12 ; i++ ) {
        x = velikaIvice(i, 0);
        y = velikaIvice(i, 1);
        glLineWidth(3.0f);
        glBegin(GL_LINES);
            glColor3f(1, 0, 0);
            glVertex3f(rekonstruisane_400(x, 0),rekonstruisane_400(x, 1), rekonstruisane_400(x, 2) );
            glVertex3f(rekonstruisane_400(y, 0),rekonstruisane_400(y, 1), rekonstruisane_400(y, 2));
        glEnd();
        
    }    
}

Eigen::Vector3d nevidljive_proizvod( Eigen::Vector3d & a, Eigen::Vector3d & b, Eigen::Vector3d & c, 
                                     Eigen::Vector3d & d, Eigen::Vector3d & e, Eigen::Vector3d & f, 
                                     Eigen::Vector3d & g, Eigen::Vector3d & h, Eigen::Vector3d & i, Eigen::Vector3d & j) {
    Eigen::Vector3d rezultat =  (((a.cross(b)).cross(c.cross(d))).cross(e)).cross(((f.cross(g)).cross(h.cross(i))).cross(j)) ;
   
    rezultat << rezultat(0) / rezultat(2), rezultat(1) / rezultat(2), rezultat(2) / rezultat(2);
    
    return rezultat.array().round();
}
