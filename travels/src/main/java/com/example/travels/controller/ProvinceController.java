package com.example.travels.controller;

import com.example.travels.entity.Province;
import com.example.travels.entity.Result;
import com.example.travels.service.ProvinceSeriviceImpl;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@CrossOrigin
@RequestMapping("province")
@Slf4j
public class ProvinceController {

    @Autowired
    ProvinceSeriviceImpl provinceSerivice;

    @RequestMapping("findByPage")
    public Map<String,Object> findByPages(Integer page, Integer rows){
        HashMap<String,Object> hashmap = new HashMap<>();
        //数据预赋值,默认取第一页，每页取四个行。
        page = page == null?1:page;
        rows = rows == null?4:rows;

        //分页处理
        List<Province> byPages = provinceSerivice.findByPages(page, rows);

        //计算总行数
        Integer totals = provinceSerivice.findTotals();

        //转换成页数
        Integer totalPages = totals%rows==0?totals/rows:totals/rows+1;
        //注意map需要和前端的变量名字一致。
        hashmap.put("provinces",byPages);
        hashmap.put("totalPages",totalPages);
        hashmap.put("totals",totals);
        hashmap.put("page",page);
        return hashmap;
    }


    @PostMapping("save")    //保存对应的省份信息
    //@RequestBody是将post请求中内容转为一个整体对象。@RequestBody的解析有两个条件：
    //1.POST请求中content的值必须为json格式（存储形式可以是字符串，也可以是byte数组）；
    //2.@RequestBody注解的参数类型必须是完全可以接收参数值的类型，
    public Result saveProvince(@RequestBody Province province){
        Result result = new Result();
        try {
            provinceSerivice.saveProvince(province);
            result.setMsg("保存成功!");
        }catch (Exception e){
            result.setState(false).setMsg("保存失败");
        }
        return result;
    }

    @GetMapping("delete")
    //Get不支持使用http Body获取参数，他只支持params，也就是URL拼接参数。get没有请求体
    public Result deleteProvince(String id){
        Result result = new Result();
        try {
            provinceSerivice.deleteProvince(String.valueOf(id));
            result.setState(true).setMsg("删除成功!");
        }catch (Exception e){
            result.setState(false).setMsg("删除失败");
        }
        return result;
    }

    //修改省份：省份信息回显，查询对应的单个省份信息
    @GetMapping("findOne")
    public Province findOneProvince(String id){
        //找到一个对应的省份信息
        Province province = null;
        try {
            province = provinceSerivice.findOne(id);
        }catch (Exception e){
            e.printStackTrace();
        }
        if (province == null){
            throw new RuntimeException("没有找到该省份");
        }
        return province;
    }

    //修改省份信息
    @PostMapping("updateProvince")
    public Result updateProvince(@RequestBody Province province){
        //前端传回的是我们之前查询到的省份信息+客户修改的部分信息。
        Result result = new Result();
        try {
            provinceSerivice.updateProvince(province);
            result.setMsg("更改省份信息成功");
        }catch (Exception e){
            result.setMsg("更改失败").setState(false);
            e.printStackTrace();
        }
        return result;
    }
}
