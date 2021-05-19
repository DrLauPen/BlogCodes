package com.example.travels.controller;

import com.example.travels.entity.Place;
import com.example.travels.entity.Result;
import com.example.travels.service.PlaceServiceImpl;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FilenameUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.util.Base64Utils;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("place")
@CrossOrigin    //允许跨域
@Slf4j
public class PlaceController {

    @Autowired
    private PlaceServiceImpl placeService;

    @Value("${upload.dir}") // @Value对值进行注入
    private String realPath;

    //对相应的景点的信息进行更新
    @PostMapping("update")
    public Result updatePlace(MultipartFile pic,Place place){
        Result result = new Result();
        try {
            //将图片转黄成base64的格式.
            String picpath  = Base64Utils.encodeToString(pic.getBytes());
            place.setPicpath(picpath);
            // 更新对应的景点信息.
            String extension = FilenameUtils.getExtension(pic.getOriginalFilename());
            //文件上传
            String newFileName  = new SimpleDateFormat("yyyyMMddHHmmss").format(new Date())+extension;
            pic.transferTo(new File(realPath,newFileName)); // 可以理解成传送到对应的浏览器中.

            placeService.updatePlace(place);
            result.setState(true).setMsg("修改成功");
        }catch (Exception e){
            e.printStackTrace();
            result.setMsg("修改失败").setState(false);
        }
        return result;
    }

    //查询一个景点的信息
    @GetMapping("findOne")
    public Place findOnePlace(String id){
        Place onePlace;
        try {
            onePlace = placeService.findOnePlace(id);
        }catch (Exception e){
            throw new RuntimeException("查找失败!");
        }
        return onePlace;
    }

    //删除对应景点信息
    @GetMapping("delete")
    public Result deletePlace(String id){
        //删除对应的景点信息
        Result result = new Result();
        try {
            placeService.deletePlace(id);
            result.setMsg("删除成功").setState(true);
        }catch (Exception e){
            result.setMsg("删除失败").setState(false);
        }
        return result;
    }

    //  保存景点信息
    @PostMapping("save")
    public Result save(MultipartFile pic, Place place) throws IOException {
        Result result = new Result();
        try {
            //先进行base64对图片进行编码
            place.setPicpath(Base64Utils.encodeToString(pic.getBytes()));
//            log.info(place.getPicpath()+"===============");
            //获取文件的拓展名.png啥的
            String extension = FilenameUtils.getExtension(pic.getOriginalFilename());
            //文件上传
            String newFileName  = new SimpleDateFormat("yyyyMMddHHmmss").format(new Date())+extension;
            pic.transferTo(new File(realPath,newFileName));

            //保存place对象
            placeService.save(place);
            result.setState(true).setMsg("保存景点信息成功");
        }catch (Exception e){
            result.setMsg(e.getMessage()).setState(false);
        }
        return result;
    }

    //根据省份查询景点的方法,与教程不同这里只给了provinceID？
    @GetMapping("findAllPlaces")
    public Map<String,Object> findAllPlaces(Integer page,Integer rows,String provinceId){
        //默认赋值
        page = page ==null?1:page;
        rows = rows==null?2:rows;
        Map<String,Object> result = new HashMap<>();

        //当前页面的景点集合
        List<Place> places = placeService.findByProvinceIdPages(page, rows, provinceId);

        //找到所有的省份景点的总数
        Integer counts = placeService.findByProvinceIdCounts(provinceId);

        //计算总页面数
        Integer totalPages = counts%rows == 0?counts/rows:counts/rows+1;

        //保存到最后的结果中，并返回到前端页面
        result.put("places",places);
        result.put("counts",counts);
        result.put("totalPages",totalPages);
        result.put("page",page);
//        log.info(page+"+=====+"+result.toString());
        return result;
    }
}
