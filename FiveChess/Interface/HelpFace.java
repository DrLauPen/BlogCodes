package Interface;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.*;

public class HelpFace {
	JDialog sframe;
	JTextPane text;
	JButton closeButton;
	
	HelpFace()
	{
		sframe=new JDialog();
		sframe.setTitle("帮助");
		sframe.setSize(630,480);
		sframe.setLayout(new BorderLayout());
		text=new JTextPane();
		text.setEditable(false);
		text.setText("(1)对局双方各执一色棋子。\r\n\n" + 
				"(2)空棋盘开局。\r\n\n" + 
				"(3)黑先、白后，交替下子，每次只能下一子。\r\n\n" + 
				"(4)棋子下在棋盘的空白点上，棋子下定后，不得向其它点移动，不得从棋盘上拿掉或拿起另落别处。\r\n\n" + 
				"(5)黑方的第一枚棋子可下在棋盘任意交叉点上。\r\n\n" + 
				"(6)轮流下子是双方的权利，但允许任何一方放弃下子权（即：PASS权）");
		Font f=new Font("隶书",Font.BOLD,20);
		text.setFont(f);
		text.setBackground(new Color(228,243,184));
		closeButton = new JButton("关闭");
		closeButton.setBackground(new Color(246,236,197));
		JPanel pa = new JPanel();
		pa.add(closeButton);
		pa.setSize(sframe.getSize());
		pa.setBackground(new Color(228,243,184));
		
		sframe.add(text,BorderLayout.CENTER);
		sframe.add(pa,BorderLayout.SOUTH);
		
		closeButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent arg0) {
				// TODO 自动生成的方法存根
				sframe.dispose();
			}
		});
		
		sframe.setModal(true);
		sframe.setLocationRelativeTo(null);
		sframe.setVisible(true);
		sframe.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
	}
}
