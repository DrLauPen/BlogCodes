  package Interface;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.Vector;

import javax.swing.*;
import javax.swing.border.Border;

public class OnLineFace implements ActionListener{
	JDialog onDialog;
	JButton leaveButton;
	JButton reButton;
	JButton losButton;
	int timeRe;
	DrawPanel chessPanel;
	
	public OnLineFace() {
		timeRe = 3;
		onDialog = new JDialog();
		onDialog.setTitle("联机模式");
		onDialog.setSize(900, 680);
		onDialog.setLayout(null);
		JPanel mPanel = new JPanel();
		mPanel.setSize(onDialog.getSize());
		mPanel.setBackground(new Color(228,243,184));
		
		leaveButton = new JButton("离开");
		reButton = new JButton("悔棋(3)");
		losButton = new JButton("认输");
		this.setButton(leaveButton);
		this.setButton(reButton);
		this.setButton(losButton);
		
		JPanel panel= new JPanel();
		panel.setBounds(60, 220, 125, 180);
		panel.setLayout(new GridLayout(3,1));
		panel.add(leaveButton);
		panel.add(reButton);
		panel.add(losButton);
		
		
		chessPanel = new DrawPanel();
		chessPanel.setBounds(265,60,530,530);
		//chessPanel.setBackground(new Color(243,221,116));
		chessPanel.setBackground(new Color(216,202,175));
		chessPanel.addMouseListener(new MouseAdapter() {
			public void mousePressed(MouseEvent e) {
				if(e.getButton() == 1) {  //---------单击左键
					int x1 = chessPanel.getStandardPoint(e.getX());
					int y1 = chessPanel.getStandardPoint(e.getY());
					if((x1 < 15 && y1 < 15) && chessPanel.chessInfo[x1][y1] == 0 ) {
						/*chessPanel.clickTimes += 1;
						Color followColor = Color.BLACK; //上一步棋颜色
						Graphics g = chessPanel.getGraphics();
						if(chessPanel.clickTimes % 2 == 1) {
							chessPanel.drawChess(x1, y1,Color.BLACK , g);
							followColor = Color.WHITE;
							chessPanel.chessInfo[x1][y1] = 0;
						}else {
							chessPanel.drawChess(x1, y1, Color.WHITE, g);
							chessPanel.chessInfo[x1][y1] = 1;
						}
						//----------------重绘上一步棋子
						int x2=0,y2=0;
						if(chessPanel.stepVe.size() != 0) {
							Vector ac = (Vector)chessPanel.stepVe.lastElement();
							x2 = (int)ac.elementAt(0);
							y2 = (int)ac.elementAt(1);
							chessPanel.clearSpot(x2,y2,followColor, g);
						}*/
			
						Vector ve = new Vector();
						ve.add(x1);
						ve.add(y1);
						chessPanel.stepVe.add(ve);
					}
				}
			}
		});
		onDialog.add(chessPanel);
		
		onDialog.add(panel);
		onDialog.add(mPanel);
		onDialog.setModal(true);
		onDialog.setResizable(false);
		onDialog.setLocationRelativeTo(null);
		onDialog.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		
		leaveButton.addActionListener(this);
		reButton.addActionListener(this);
		losButton.addActionListener(this);
		onDialog.setVisible(true);
		
	}

		@Override
		public void actionPerformed(ActionEvent e) {
			// TODO 自动生成的方法存根
			if(e.getSource() == leaveButton) {
				onDialog.dispose();
			}else if(e.getSource() == reButton) {
				if(this.timeRe > 0) {
					Graphics g = chessPanel.getGraphics();
					chessPanel.regretStep(g);
					this.timeRe -= 1;
					reButton.setText("悔棋("+String.valueOf(timeRe)+")");
				}else {
					JOptionPane.showMessageDialog(onDialog, "无悔棋机会!","警告",JOptionPane.INFORMATION_MESSAGE);
				}
			}else if(e.getSource() == losButton) {
				JOptionPane.showMessageDialog(onDialog, "你输了...","警告",JOptionPane.INFORMATION_MESSAGE);
			}else {
				
			}
		}
	
		public void setButton(JButton bu) {
			Border buttonBorder = BorderFactory.createLineBorder(Color.WHITE, 4);
			bu.setFont(new Font("隶书",Font.BOLD,24));
			bu.setBorder(buttonBorder);
			bu.setBackground(new Color(246,236,197));
		}
		
}
