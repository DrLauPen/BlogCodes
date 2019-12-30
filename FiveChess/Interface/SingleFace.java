package Interface;
import java.awt.*;
import java.awt.event.*;
import java.util.Vector;

import javax.swing.*;
import javax.swing.border.Border;

import Algorithm.Receive;

public class SingleFace implements ActionListener{
	JDialog sDialog;
	JButton menuButton;
	JButton newButton;
	JButton setButton;
	JButton regretButton;
	JButton loseButton;
	DrawPanel chessPanel;
	JLabel leverLabel;
	JLabel selectLabel;
	JPanel panel;
	int robotLever;    //机器难度等级
	
	JLabel passLabel;
	JComboBox passBox;
	int passItem;
	int ynLast  ;   //ynLast为0时是单人模式界面，为1时是残局模式	
	int basicNum;   //残局初始步数
	
	public SingleFace() {
		
	}
	
	public SingleFace(int ynSL) {
		//ynSL0时是单人模式界面，为1时是残局模式
		robotLever = 1;   //默认难度为1
		this.ynLast = ynSL;
		sDialog = new JDialog();
		sDialog.setTitle("单机模式");
		sDialog.setSize(900,680);
		sDialog.setLayout(null);
		
		JPanel mPanel = new JPanel();
		mPanel.setSize(sDialog.getSize());
		mPanel.setBackground(new Color(228,243,184));
	    //mPanel.setBackground(new Color(246,236,197));
		
		menuButton = new JButton("菜单");
		newButton  = new JButton("新游戏");
		regretButton = new JButton("悔棋");
		loseButton = new JButton("认输");
		panel= new JPanel();
		panel.setBounds(60, 220, 125, 180);
		panel.setLayout(new GridLayout(5,1));
		panel.add(menuButton);
		panel.add(newButton);
		if(ynSL == 0) {
			this.setUpdate();
		}else {
			this.passUpdate();
		}

		panel.add(regretButton);
		panel.add(loseButton);
			
		JPanel p1 = new JPanel();
		p1.setBounds(60,440,125,80);
		p1.setLayout(new GridLayout(2,1));
		selectLabel = new JLabel("您的棋子:●",SwingConstants.CENTER);
		selectLabel.setFont(new Font("隶书",Font.BOLD,18));
		selectLabel.setBackground(new Color(246,236,197));
		p1.add(selectLabel);
		p1.add(leverLabel);
		p1.setBackground(new Color(246,236,197));
		
		//--------------------棋盘
		chessPanel = new DrawPanel();
		chessPanel.setBounds(265,60,530,530);
		chessPanel.setBackground(new Color(216,202,175));
		chessPanel.addMouseListener(new MouseAdapter() {
			public void mousePressed(MouseEvent e) {
				if(e.getButton() == 1) {  //---------单击左键
					int x1 = chessPanel.getStandardPoint(e.getX());
					int y1 = chessPanel.getStandardPoint(e.getY());
					if((x1 < 15 && y1 < 15) && chessPanel.chessInfo[x1][y1] == 0 ) {
						if(ynLast == 0) {
							if(chessPanel.chessColor == Color.BLACK) {
								blackPlayer(x1,y1);
							}else {
								whitePlayer(x1,y1);
							}
						}else {
							blackPlayer(x1,y1);
						}
					}
				}
			}
		});
		
		this.setButton(menuButton);
		this.setButton(newButton);
		this.setButton(regretButton);
		this.setButton(loseButton);
		
		sDialog.add(p1);
		sDialog.add(chessPanel);
		sDialog.add(panel);
		sDialog.add(mPanel);
		sDialog.setModal(true);
		sDialog.setResizable(false);
		sDialog.setLocationRelativeTo(null);
		sDialog.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		
		menuButton.addActionListener(this);
		newButton.addActionListener(this);
		regretButton.addActionListener(this);
		loseButton.addActionListener(this);
		sDialog.setVisible(true);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		// TODO 自动生成的方法存根
		if(e.getSource() == menuButton) {
			sDialog.dispose();
		}else if(e.getSource() == newButton) {
			this.newGame();
		}else if(e.getSource() == regretButton) {
			if(ynLast == 0) {
				Graphics g = chessPanel.getGraphics();
				chessPanel.regretStep(g);
				chessPanel.regretStep(g);
			}else {
				if(chessPanel.stepVe.size() > basicNum) {
					Graphics g = chessPanel.getGraphics();
				    chessPanel.regretStep(g);
				}
			}
		
		}else if(e.getSource() == loseButton) {
			JOptionPane.showMessageDialog(sDialog, "你输了...","警告",JOptionPane.INFORMATION_MESSAGE);
		}else {
			
		}
	}
	
	//setButton人机模式设置
	public void setUpdate() {
		leverLabel  = new JLabel("难度等级:★",SwingConstants.CENTER);
		leverLabel.setBackground(new Color(246,236,197));
		leverLabel.setFont(new Font("隶书",Font.BOLD,16));
		setButton = new JButton("设置");
		this.setButton(setButton);
		panel.add(setButton);
		setButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				// TODO 自动生成的方法存根
				if(chessPanel.stepVe.size() == 0) {
					SingleSetFa setface = new SingleSetFa();
				    if(setface.selectChess == 1) {
				    	chessPanel.chessColor = Color.BLACK;
					    selectLabel.setText("您的棋子 ●");
				    }else {
					    chessPanel.chessColor = Color.white;
					    selectLabel.setText("您的棋子 ○");
					    //chessPanel.clickTimes += 1;
			     	}
				    String stra = new String("★");
				    for(int i=1;i<setface.lever;i++)
				    	stra += "★";
				    leverLabel.setText("难度等级"+stra);
				    robotLever = setface.lever;
				}else {
					JOptionPane.showMessageDialog(sDialog, "游戏正在进行中...","警告",JOptionPane.INFORMATION_MESSAGE);
				}
			}
		});
	}
	
	//passButton残局模式关卡选择
	public void passUpdate() {
		passItem = 1;    //默认为第一关
		basicNum = 12;
		passLabel = new JLabel();
		passLabel.setText("关卡");
		passLabel.setFont(new Font("隶书",Font.BOLD,18));
		passLabel.setBackground(new Color(246,236,197));
		JPanel pa = new JPanel();
		pa.setBackground(new Color(246,236,197));
		pa.setLayout(new GridLayout(1,2));
		String pStr[] = {"No.1","No.2","No.3","No.4","No.5"};
		passBox = new JComboBox(pStr);
		passBox.setBackground(new Color(246,236,197));
		pa.add(passLabel);
		pa.add(passBox);
		panel.add(pa);
		leverLabel  = new JLabel("关卡   No1",SwingConstants.CENTER);
		leverLabel.setBackground(new Color(246,236,197));
		leverLabel.setFont(new Font("隶书",Font.BOLD,16));
		passBox.addItemListener(new ItemListener() {
			@Override
			public void itemStateChanged(ItemEvent e) {
				// TODO 自动生成的方法存根
				passItem = passBox.getSelectedIndex()+1;
				leverLabel.setText("关卡  No"+Integer.toString(passItem));
				newGame();
				drawCM(passItem);
			}
		});
	}
	
	//新游戏
	public void newGame() {
		if(this.ynLast == 1) {
			Graphics g = chessPanel.getGraphics();
			for(int i=0;i<chessPanel.stepVe.size();i++) {
				chessPanel.regretStep(g);
			}
			for(int i=0;i<15;i++) {
				for(int j=0;j<15;j++)
					chessPanel.chessInfo[i][j] = 0;
			}
			chessPanel.stepVe.clear();
			drawCM(passItem);
		}else {
			chessPanel.repaint();
			for(int i=0;i<15;i++) {
				for(int j=0;j<15;j++)
					chessPanel.chessInfo[i][j] = 0;
			}
			chessPanel.stepVe.clear();
		}
	}
	
	//模式棋谱绘制
	public void drawCM(int flag) {
		CManual cm = new CManual();
		cm.choose(flag);
		chessPanel.stepVe = cm.chessVe;
		basicNum = chessPanel.stepVe.size();
		Graphics g = chessPanel.getGraphics();
		for(int i=0;i<cm.chessVe.size();i++) {
			Vector ve = (Vector) cm.chessVe.elementAt(i);
			int x1 = (int)ve.elementAt(0);
			int y1 = (int)ve.elementAt(1);
			chessPanel.clickTimes += 1;
			if(chessPanel.clickTimes % 2 == 1) {
				chessPanel.clearSpot(x1, y1,Color.BLACK , g);
				chessPanel.chessInfo[x1][y1] = 1;
			}else {
				chessPanel.clearSpot(x1, y1, Color.WHITE, g);
				chessPanel.chessInfo[x1][y1] = 2;
			}
		}
	}
	
	//按钮风格设置
	public void setButton(JButton bu) {
		Border buttonBorder = BorderFactory.createLineBorder(Color.WHITE, 2);
		bu.setFont(new Font("隶书",Font.BOLD,18));
		bu.setBorder(buttonBorder);
		bu.setBackground(new Color(246,236,197));
	}
	
	//玩家为黑棋
	public void blackPlayer(int x1,int y1) {
		Graphics g = chessPanel.getGraphics();
		Vector ve = new Vector();     //---------------取消AI标记
		if(chessPanel.stepVe.size()> 1) {
			ve = (Vector)chessPanel.stepVe.elementAt(chessPanel.stepVe.size()-1);
			int x2 = (int)ve.elementAt(0);
			int y2 = (int)ve.elementAt(1);
			chessPanel.clearSpot(x2,y2,Color.WHITE, g);
		}
		
		chessPanel.drawChess(x1,y1,Color.BLACK,g);//----------人
		chessPanel.chessInfo[x1][y1] = 1;
		ve = new Vector();  
		ve.add(x1);
		ve.add(y1);
		chessPanel.stepVe.add(ve);
		
		int []point = new int[3];
		if(ynLast == 0) {
			point = Receive.Receive(robotLever, chessPanel.chessInfo, x1, y1);
		}else {
			point = Receive.Receive(3, chessPanel.chessInfo, x1, y1);
		}
		if(point[2] == 1) {
			JOptionPane.showMessageDialog(sDialog, "你赢了...","警告",JOptionPane.INFORMATION_MESSAGE);
		}else {
			int []point2 = new int[3];
			if(ynLast == 0) {
				point2= Receive.Receive(robotLever, chessPanel.chessInfo, point[0], point[1]);
			}else {
				point2= Receive.Receive(3, chessPanel.chessInfo, point[0], point[1]);
			}
			chessPanel.clearSpot(x1,y1,Color.BLACK,g);//----------取消人标记
			chessPanel.drawChess(point[0],point[1], Color.WHITE, g);  //----AI
			chessPanel.chessInfo[point[0]][point[1]] = 2;
			ve = new Vector();
			ve.add(point[0]);
			ve.add(point[1]);
			chessPanel.stepVe.add(ve);
			if(point[2] == 2) {
				JOptionPane.showMessageDialog(sDialog, "你输了...","警告",JOptionPane.INFORMATION_MESSAGE);
			}
			
		}
	}
	
	//玩家为白棋
	public void whitePlayer(int x1,int y1) {
		Graphics g = chessPanel.getGraphics();
		Vector ve = new Vector();     //---------------取消AI标记
		if(chessPanel.stepVe.size()> 1) {
			ve = (Vector)chessPanel.stepVe.elementAt(chessPanel.stepVe.size()-1);
			int x2 = (int)ve.elementAt(0);
			int y2 = (int)ve.elementAt(1);
			chessPanel.clearSpot(x2,y2,Color.BLACK, g);
		}
		
		chessPanel.drawChess(x1,y1,Color.WHITE,g);//----------人
		chessPanel.chessInfo[x1][y1] = 1;
		ve = new Vector();  
		ve.add(x1);
		ve.add(y1);
		chessPanel.stepVe.add(ve);
		
		int []point = new int[3];
		point = Receive.Receive(robotLever, chessPanel.chessInfo, x1, y1);
		if(point[2] == 1) {
			JOptionPane.showMessageDialog(sDialog, "你赢了...","警告",JOptionPane.INFORMATION_MESSAGE);
		}else {
			int []point2 = new int[3];
			point2= Receive.Receive(robotLever, chessPanel.chessInfo, point[0], point[1]);
			chessPanel.clearSpot(x1,y1,Color.WHITE,g);//----------取消人标记
			chessPanel.drawChess(point[0],point[1], Color.BLACK, g);  //----AI
			chessPanel.chessInfo[point[0]][point[1]] = 2;
			ve = new Vector();
			ve.add(point[0]);
			ve.add(point[1]);
			chessPanel.stepVe.add(ve);
			if(point[2] == 2) {
				JOptionPane.showMessageDialog(sDialog, "你输了...","警告",JOptionPane.INFORMATION_MESSAGE);
			}
		}
	}
	
}


