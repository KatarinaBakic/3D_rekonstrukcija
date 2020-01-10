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

MatrixXd rekonstruisane_400(16, 3);
        
MatrixXd izracunavanje();
MatrixXd uAfine(MatrixXd &xx);
MatrixXd triD(MatrixXd &xx, MatrixXd &yy, MatrixXd &t1, MatrixXd &t2);
void nacrtaj_malu( MatrixXd &rekonstruisane_400);
void nacrtaj_veliku( MatrixXd &rekonstruisane_400);

Eigen::Vector3d nevidljive_proizvod( Eigen::Vector3d & a, Eigen::Vector3d & b, Eigen::Vector3d & c, 
                                     Eigen::Vector3d & d, Eigen::Vector3d & e, Eigen::Vector3d & f, 
                                     Eigen::Vector3d & g, Eigen::Vector3d & h, Eigen::Vector3d & i, Eigen::Vector3d & j);

    
void nacrtaj_ose();
int main (int argc, char * argv[]){
    
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(1000, 1000);
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

    gluLookAt(50, 75, 25,
               0,  0,  0,
               0,  0,  1
             );

    //nacrtaj_ose();
    
    if( i == 0 ){
        rekonstruisane_400 << izracunavanje()*0.05;
        std::cout << "Rekonstruisane400 : \n" << rekonstruisane_400 << std::endl; 
        i++;
    }
    glPushMatrix();
        nacrtaj_malu(rekonstruisane_400);
        nacrtaj_veliku(rekonstruisane_400);
    glPopMatrix();
    glutSwapBuffers();
}



MatrixXd izracunavanje() {

    MatrixXd x1(1, 3), x2(1, 3), x3(1, 3), x4(1, 3), x9(1, 3), x10(1, 3), x11(1, 3), x12(1, 3);
    MatrixXd y1(1, 3), y2(1, 3), y3(1, 3), y4(1, 3), y9(1, 3), y10(1, 3), y11(1, 3), y12(1, 3);
  
    // osam tacaka za fundamentalnu matricu F = FF

    x1  << 958,  38,   1;     
    y1  << 933,  33,   1;    
    x2  << 1117, 111,  1;
    y2  << 1027, 132,  1;
    x3  << 874,  285,  1;
    y3  << 692,  223,  1;
    x4  << 707,  218,  1;
    y4  << 595,  123,  1;
    x9  << 292,  569,  1;
    y9  << 272,  360,  1;    
    x10 << 770,  969,  1;
    y10 << 432,  814,  1;
    x11 << 770,  1465, 1;
    y11 << 414,  1284, 1;    
    x12 << 317,  1057, 1;    
    y12 << 258,  818,  1;
    

    MatrixXd xx(8, 3), yy(8, 3);
       
    xx<< x1, x2, x3, x4, x9, x10, x11, x12 ;
    yy<< y1, y2, y3, y4, y9, y10, y11, y12 ;
    
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
    // vektori skrivenih tacaka:     
    Eigen::Vector3d x_5, x_6, x_7, x_8, x_13, x_14, x_15, x_16;
    Eigen::Vector3d y_5, y_6, y_7, y_8, y_13, y_14, y_15, y_16;
    
    // potrebne su nam i prethodne tacke kao vektori, zbog vektorskog mnozenja! 

    Eigen::Vector3d x_1, x_2, x_3, x_4, x_9, x_10, x_11, x_12;
    Eigen::Vector3d y_1, y_2, y_3, y_4, y_9, y_10, y_11, y_12;
    
    x_1  << x1(0),  x1(1), x1(2);
    y_1  << y1(0),  y1(1), y1(2);
    x_2  << x2(0),  x2(1), x2(2);
    y_2  << y2(0),  y2(1), y2(2);
    x_3  << x3(0),  x3(1), x3(2);
    y_3  << y3(0),  y3(1), y3(2);
    x_4  << x4(0),  x4(1), x4(2);
    y_4  << y4(0),  y4(1), y4(2);
    x_9  << x9(0),  x9(1), x9(2);
    y_9  << y9(0),  y9(1), y9(2);
    x_10 << x10(0), x10(1),x10(2);
    y_10 << y10(0), y10(1),y10(2);
    x_11 << x11(0), x11(1),x11(2);
    y_11 << y11(0), y11(1),y11(2);
    x_12 << x12(0), x12(1),x12(2);
    y_12 << y12(0), y12(1),y12(2);

    x_6 << 1094, 536, 1;
    y_6 << 980, 535, 1;
    x_7 << 862, 729, 1;
    y_7 << 652, 638, 1;
    x_8 << 710, 648, 1;
    y_8 << 567,532, 1;
    x_14 << 1487, 598, 1;
    y_14 << 1303, 700, 1;
    x_15 << 1462, 1079, 1;
    y_15 << 1257, 1165, 1;
    y_13 << 1077, 269, 1;
    
    x_5 = nevidljive_proizvod ( x_4, x_8, x_6, x_2, x_1,
                                x_1, x_4, x_3, x_2, x_8 );
     
    y_5 = nevidljive_proizvod ( y_4, y_8, y_6, y_2, y_1, 
                                y_1, y_4, y_3, y_2, y_8 );
    
    // std::cout << "X5 :" << x_5 <<"\n";
    
    x_13 = nevidljive_proizvod(  x_9, x_10, x_11, x_12, x_14,
                                x_11, x_15, x_10, x_14,  x_9 );
    
    
    x_16 = nevidljive_proizvod( x_10, x_14, x_11, x_15, x_12,
                                 x_9, x_10, x_11, x_12, x_15 );
    
    y_16 = nevidljive_proizvod( y_10, y_14, y_11, y_15, y_12, 
                                 y_9, y_10, y_11, y_12, y_15 );
    
    //potreban nam je dalje rad sa matricama: 
    MatrixXd x5(1, 3), x6(1, 3), x7(1, 3), x8(1, 3), x13(1, 3), x14(1, 3), x15(1, 3), x16(1, 3);
    MatrixXd y5(1, 3), y6(1, 3), y7(1, 3), y8(1, 3), y13(1, 3), y14(1, 3), y15(1, 3), y16(1, 3);
    
    x5 <<  x_5(0),  x_5(1),  x_5(2)  ;
    x6 <<  x_6(0),  x_6(1),  x_6(2)  ;
    x7 <<  x_7(0),  x_7(1),  x_7(2)  ;
    x8 <<  x_8(0),  x_8(1),  x_8(2)  ;
    x13 << x_13(0), x_13(1), x_13(2) ;
    x14 << x_14(0), x_14(1), x_14(2) ;
    x15 << x_15(0), x_15(1), x_15(2) ;
    x16 << x_16(0), x_16(1), x_16(2) ;
    
    y5  << y_5(0),  y_5(1),  y_5(2)  ; 
    y6  << y_6(0),  y_6(1),  y_6(2)  ;
    y7  << y_7(0),  y_7(1),  y_7(2)  ;
    y8  << y_8(0),  y_8(1),  y_8(2)  ;
    y13 << y_13(0), y_13(1), y_13(2) ;
    y14 << y_14(0), y_14(1), y_14(2) ;
    y15 << y_15(0), y_15(1), y_15(2) ;
    y16 << y_16(0), y_16(1), y_16(2) ;
    
    
    
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
    MatrixXd slika1(16, 3), slika2(16, 3) ;
    slika1 << x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 ;
    slika2 << y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16 ;
   
    MatrixXd jednacine(16, 4), prva_pom(1, 3), druga_pom(1, 3);
    for(int i = 0; i < 16 ; i++){
        prva_pom << slika1.row(i);
        druga_pom << slika2. row(i);
        jednacine.row(i) << triD (prva_pom, druga_pom, t1, t2);
    }
    
    // prebacujemo koordinate u afine :
    MatrixXd rekonstruisane(16, 3), pom(1, 4);
    for (int i = 0; i < 16; i++){
        pom << jednacine.row(i);
        rekonstruisane.row(i) << uAfine(pom); 
    }
    
    //std::cout << rekonstruisane << "\n" ;
    //mnozimo sa nekom stotinom (npr. 400 ) poslednju koordinatu, zato sto je jako mala u odnosu na ostale: 

    MatrixXd rekonstruisane400(16, 3);
    
    rekonstruisane400 << rekonstruisane.col(0),rekonstruisane.col(1), rekonstruisane.col(2)*400;
    
    //std::cout << "Rekonstruisane400 :\n" << rekonstruisane400 << "\n" << std::endl ;
 
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
            glColor3f(0, 0.4, 0.4);
            glVertex3f(rekonstruisane_400(x, 0),rekonstruisane_400(x, 1), rekonstruisane_400(x, 2) );
            glVertex3f(rekonstruisane_400(y, 0),rekonstruisane_400(y, 1), rekonstruisane_400(y, 2));
        glEnd();
        
    }    
}

/*velika kutija */                 
void nacrtaj_veliku( MatrixXd &rekonstruisane_400) {
    
    MatrixXd velikaIvice (12, 2);
    velikaIvice <<  8, 9,
                  9, 10,
                 10, 11,
                 11, 8,
                 12, 13,
                 13, 14,
                 14, 15,
                 15, 12,
                  8, 12,
                  9, 13,
                 10, 14,
                 11, 15;
    
    float x, y;
    for ( int i = 0 ; i < 12 ; i++ ) {
        x = velikaIvice(i, 0);
        y = velikaIvice(i, 1);
        glBegin(GL_LINES);
            glColor3f(1, 0.4, 0.4);
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
