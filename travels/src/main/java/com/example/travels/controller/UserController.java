package com.example.travels.controller;

import com.example.travels.entity.Result;
import com.example.travels.entity.User;
import com.example.travels.service.UserServiceImpl;
import com.example.travels.utils.CreateImageCode;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.Base64Utils;
import org.springframework.web.bind.annotation.*;

import javax.imageio.ImageIO;
import javax.servlet.http.HttpServletRequest;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("user")
@CrossOrigin    //允许跨域
@Slf4j
public class UserController {

    @Autowired
    UserServiceImpl userService;

    //用户登录
    @RequestMapping("login")
    public Result login(@RequestBody User user,HttpServletRequest request){
        Result result = new Result();
        log.info(user.toString()+"----------------------");
        try{
            User userDB = userService.login(user);
            result.setMsg("登录成功");
            result.setState(true);
            //如果登录成功保存用户的登录信息.目前采用context进行做的，需要采用redis进行改进
            request.getServletContext().setAttribute(userDB.getId(),user);
        }catch (Exception e){
            result.setMsg(e.getMessage());
            result.setState(false);
        }
        return result;
    }

    //用户注册
    @PostMapping("register")
    public Result register(String code,String key,@RequestBody User user, HttpServletRequest httpServletRequest){
        log.info("接受的验证码"+code);
        log.info("接受到的user对象"+user);
        //验证验证码
        Result result = new Result();
        String keycode = (String) httpServletRequest.getServletContext().getAttribute(key);
        log.info(keycode);
        //注册用户
        try{
            if(keycode.equalsIgnoreCase(code)) {//验证码匹配,不区分大小写
                userService.register(user);//需要对用户进行校验确保不存在相同的用户
                result.setMsg("注册成功!");//还需要跳转到对应的页面
                result.setState(true); //设置让对应的页面跳转
            }else {
                throw new RuntimeException("验证码验证失败。");
            }
        }catch (Exception e){
            e.printStackTrace();
            result.setMsg(e.getMessage());
            result.setState(false);
        }
        return result;
    }

    //提供验证码
    @GetMapping("getImage")
    @ResponseBody //将java对象转为json格式的数据。
    public Map<String,String> getImage(HttpServletRequest httpServletRequest) throws IOException {
        Map<String,String> result = new HashMap<>();

        //获取验证码
        CreateImageCode vCode = new CreateImageCode(100, 30, 5, 10);
        String securityCode= vCode.getCode();
        // 存入session不可行！每次会话都是一个新的 保存到servletcontext中
        String key  = new SimpleDateFormat("yyyyMMddHHmmss").format(new Date());
        httpServletRequest.getServletContext().setAttribute(key,securityCode);

        //生成对应的图片并进行base64的编码
        BufferedImage bufferedImage = vCode.getBuffImg();
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        ImageIO.write(bufferedImage,"png",bos);
        String string = Base64Utils.encodeToString(bos.toByteArray());

        //保存到map中
        result.put("key",key);
        result.put("image",string);

        // 返回浏览器
        return result;
    }
}
