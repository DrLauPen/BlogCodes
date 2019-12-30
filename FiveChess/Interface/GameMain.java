package Interface;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.Border;

class MajorFace implements ActionListener{
	JFrame frame;
	JButton singleButton;
	JButton onlineButton;
	JButton brokenButton;
	JButton helpButton;
	JButton endButton;      
	
	
	public MajorFace(){
		frame = new JFrame("五子棋");
		frame.setSize(600, 480);
		frame.setLayout(null);
		frame.setLocationRelativeTo(null);
		frame.setResizable(false);
		JPanel mPanel = new JPanel();
		mPanel.setSize(frame.getSize());
		mPanel.setBackground(new Color(228,243,184));
		
		
		singleButton = new JButton("单机模式");
		onlineButton = new JButton("联机模式");
		brokenButton = new JButton("残局模式");
		helpButton = new JButton("游戏帮助");
		endButton  = new JButton("退出游戏");
		
		this.setButton(singleButton);
		this.setButton(onlineButton);
		this.setButton(brokenButton);
		this.setButton(helpButton);
		this.setButton(endButton);
		
		JTextPane tPane = new JTextPane();
		tPane.setBounds(130, 40, 160, 360);
		tPane.setEditable(false);
		tPane.setFont(new Font("隶书",Font.BOLD,100));
		tPane.setText("五子棋");
		tPane.setForeground(new Color(210,201,156));
		tPane.setBackground(new Color(228,243,184));
		
		JPanel panel = new JPanel();
		panel.setBounds(360,100,140,280);
		panel.setLayout(new GridLayout(5,1));
		panel.add(singleButton);
		panel.add(onlineButton);
		panel.add(brokenButton);
		panel.add(helpButton);
		panel.add(endButton);
		
		frame.add(tPane);
		frame.add(panel);
		frame.add(mPanel);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);
		
		singleButton.addActionListener(this);
		onlineButton.addActionListener(this);
		brokenButton.addActionListener(this);
		helpButton.addActionListener(this);
		endButton.addActionListener(this);
	
 	}
	
	@Override
	public void actionPerformed(ActionEvent e) {
		// TODO 自动生成的方法存根
		if(e.getSource() == singleButton) {
			SingleFace sf = new SingleFace(0);
			sf.sDialog.dispose();
		}else if(e.getSource() == onlineButton) {
			
			LoginFace lof= new LoginFace();
			if(lof.yn) {
				new RoomFace();
			}
			//OnLineFace of = new OnLineFace();
			
		}else if(e.getSource() == brokenButton) {
			SingleFace sfa = new SingleFace(1);
			sfa.sDialog.dispose();
		}else if(e.getSource() == helpButton) {
			HelpFace hf = new HelpFace();
		}else {
			frame.dispose();
		}
	}
	
	public void setButton(JButton bu) {
		Border buttonBorder = BorderFactory.createLineBorder(Color.WHITE, 4);
		bu.setFont(new Font("隶书",Font.BOLD,24));
		bu.setBorder(buttonBorder);
		bu.setBackground(new Color(246,236,197));
	}
}

public class GameMain {

	public static void main(String[] args) {
		// TODO 自动生成的方法存根
	    new MajorFace();
	}
}
