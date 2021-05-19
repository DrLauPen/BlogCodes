package com.example.travels.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.Accessors;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Accessors(chain = true)//使得可以.setmsg().setstate()链式调用
public class Result {
    private boolean state = true;
    private String msg;
    private int userid;//保存对应登录的userid到前台
}
