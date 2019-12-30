package Interface;
import javax.swing.*;
import java.awt.*;
import java.util.Vector;

public class DrawPanel extends JPanel{
	Color chessColor;    //棋子颜色
	int clickTimes;    //鼠标左键单击次数
	int [][]chessInfo = new int[15][15];  //棋盘数据
	Vector stepVe;    //保存下棋步骤
	
	public static final int x0 = 40;   //开始画的x值
	public static final int y0 = 40;   //开始画的y值
	public static final int ROWS = 15;   //棋盘有多少行
	public static final int COLUMNS = 15;   //棋盘有多少列
	public static final int CHESS_SIZE = 40;   //棋子的半径大小
	public static final int SIZE = 32;         //棋盘每个小方格的大小
	
	public DrawPanel() {
		chessColor = Color.BLACK;  //默认为黑色
		clickTimes = 0;
		for(int i=0;i<chessInfo.length;i++) {
			chessInfo[i] = new int[15];
			for(int j=0;j<chessInfo[i].length;j++)
				chessInfo[i][j] = 0;
		}
		stepVe = new Vector();
	}
	
	public void paint(Graphics g) {
		super.paint(g);
	    this.drawTable(g);
	}
	
	public void drawTable(Graphics g) {
		Graphics2D g2 = (Graphics2D)g;
		g2.setStroke(new BasicStroke(2.0f));
		for(int i=0;i<DrawPanel.COLUMNS;i++) {
			g2.drawLine(x0,y0+i*SIZE,(ROWS-1)*SIZE+x0,y0+i*SIZE);
			g2.drawLine(x0+i*SIZE,y0,x0+i*SIZE,y0+(COLUMNS-1)*SIZE);
		}	
		g2.fillOval(x0+3*SIZE-5, y0+3*SIZE-5, 10,10);
		g2.fillOval(x0+3*SIZE-5, y0+11*SIZE-5, 10,10);
		g2.fillOval(x0+7*SIZE-5, y0+7*SIZE-5, 10,10);
		g2.fillOval(x0+11*SIZE-5, y0+3*SIZE-5, 10,10);
		g2.fillOval(x0+11*SIZE-5, y0+11*SIZE-5, 10,10);
	}
	
	//-------------------绘制棋子
	public void drawChess(int x,int y,Color c,Graphics g) {
		g.setColor(c);
		g.fillOval(x0+x*SIZE-13, y0+y*SIZE-13, 26, 26);
		g.setColor(Color.cyan);
		g.drawLine(x0+x*SIZE-8, y0+y*SIZE, x0+x*SIZE+8,y0+y*SIZE);
		g.drawLine(x0+x*SIZE, y0+y*SIZE-8, x0+x*SIZE,y0+y*SIZE+8);
	}
	
	//-------------------重绘棋子
	public void clearSpot(int x,int y,Color c,Graphics g) {
		g.setColor(c);
		g.fillOval(x0+x*SIZE-13, y0+y*SIZE-13, 26, 26);
	}
	
	//---------坐标规范化
	public int getStandardPoint(int a) {
		int x = (a-x0)/SIZE;
		if((a-x0-x*SIZE)>13) {
			x += 1;
		}
		return x;
	}
	
	//----------------悔棋重绘棋子位置
	public void reDrawChess(int x,int y,Color c,Graphics g) {
		g.setColor(new Color(216,202,175));
		g.fillOval(x0+x*SIZE-13, y0+y*SIZE-13, 26, 26);
		Graphics2D g2 = (Graphics2D)g;
		g2.setStroke(new BasicStroke(2.0f));
		g2.setColor(Color.BLACK);
		
		// 3x3=9类坐标
		if(y == 0 && x == 0) {
			g.drawLine(x0, y0, x0+13,y0);
		    g.drawLine(x0, y0, x0,y0+13);
		}else if(y==0 && (x != 0 && x != 14)) {
			g.drawLine(x0+x*SIZE-13, y0+y*SIZE, x0+x*SIZE+13,y0+y*SIZE);
		    g.drawLine(x0+x*SIZE, y0, x0+x*SIZE,y0+y*SIZE+13);
		}else if(y==0 && x == 14) {
			g.drawLine(x0+x*SIZE-13, y0+y*SIZE, x0+x*SIZE,y0+y*SIZE);
		    g.drawLine(x0+x*SIZE, y0+y*SIZE, x0+x*SIZE,y0+y*SIZE+13);
		}else if((y != 0 && y != 14) && x == 0) {
			g.drawLine(x0+x*SIZE, y0+y*SIZE, x0+x*SIZE+13,y0+y*SIZE);
		    g.drawLine(x0+x*SIZE, y0+y*SIZE-13, x0+x*SIZE,y0+y*SIZE+13);
		}else if((y != 0 && y != 14) && (x != 0 && x != 14)) {
			g.drawLine(x0+x*SIZE-13, y0+y*SIZE, x0+x*SIZE+13,y0+y*SIZE);
		    g.drawLine(x0+x*SIZE, y0+y*SIZE-13, x0+x*SIZE,y0+y*SIZE+13);
		}else if((y != 0 && y != 14) && x == 14) {
			g.drawLine(x0+x*SIZE-13, y0+y*SIZE, x0+x*SIZE,y0+y*SIZE);
		    g.drawLine(x0+x*SIZE, y0+y*SIZE-13, x0+x*SIZE,y0+y*SIZE+13);
		}else if(y == 14 && x == 0) {
			g.drawLine(x0+x*SIZE, y0+y*SIZE, x0+x*SIZE+13,y0+y*SIZE);
		    g.drawLine(x0+x*SIZE, y0+y*SIZE-13, x0+x*SIZE,y0+y*SIZE);
		}else if(y == 14 &&  (x != 0 && x != 14)) {
			g.drawLine(x0+x*SIZE-13, y0+y*SIZE, x0+x*SIZE+13,y0+y*SIZE);
		    g.drawLine(x0+x*SIZE, y0+y*SIZE-13, x0+x*SIZE,y0+y*SIZE);
		}else {
			g.drawLine(x0+x*SIZE-13, y0+y*SIZE, x0+x*SIZE,y0+y*SIZE);
		    g.drawLine(x0+x*SIZE, y0+y*SIZE-13, x0+x*SIZE,y0+y*SIZE);
		}
	}
	
	//-------------------悔棋
	public void regretStep(Graphics g) {
		if(stepVe.size() != 0) {
			//this.repaint();
			Color c1,c2;
			if(chessColor == Color.black) {
				c1 = Color.black;
				c2 = Color.white;
			}else {
				c1 = Color.white;
				c2 = Color.black;
			}
			int x = 0 ,y = 0;
			Vector ae = (Vector)stepVe.lastElement();
			x = (int)ae.elementAt(0);
			y = (int)ae.elementAt(1);
			this.reDrawChess(x, y, c1, g);
			chessInfo[x][y] = 0;
			stepVe.remove(stepVe.size()-1);
			//clickTimes -= 1;
			
			if(stepVe.size() != 0) {
				ae = (Vector)stepVe.lastElement();
			    x = (int)ae.elementAt(0);
			    y = (int)ae.elementAt(1);
			    if((stepVe.size())%2 == 1) {
			    	this.drawChess(x, y, c1, g);
			    }else {
			    	this.drawChess(x, y, c2, g);
			    }
			 }
		}
	}
	
	
}
