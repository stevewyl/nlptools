# -*- coding: utf-8 -*-
from nlptools.preprocess import clean_text, split_text

print("Test [SPLIT]")
sent = "哈哈哈, 你好呀，嘿嘿哈哈哈哈，诶诶阿法！"
assert len(split_text(sent)) == 1
assert len(split_text(sent, cut_comma=True)) == 2
assert len(split_text(sent, cut_all=True)) == 4
sent = "虽然今天天气不错而且还发工资，但我还是很不开心因为失恋了"
assert len(split_text(sent, turn=True, coo=True)) == 3
print("All Pass [SPLIT]")

print("Test [CLEAN]")
text = "thequickbrownfoxjumpsoverthelazydog"
assert clean_text(text) == "the quick brown fox jumps over the lazy dog"
text = "arttemplate艺术模板"
assert clean_text(text) == "art template 艺术模板"
text = "回复@钱旭明QXM:[嘻嘻][嘻嘻] //@钱旭明QXM:杨大哥[good][good]"
assert clean_text(text) == "杨大哥"
text = "【#赵薇#：正筹备下一部电影 但不是青春片....http://t.cn/8FLopdQ http://www.google.com"
assert clean_text(text, weibo_topic=True) == "【正筹备下一部电影但不是青春片...."
text = "&lt;a c&gt;&nbsp;&#x27;&#x27;"
assert clean_text(text, norm_html=True) == "<ac>''"
text = "×～~"
clean_text(text, q2b=True, remove_rare_char=True) == "~~"
text = "http://study.163.com/course/courseMain.htm?courseId=1459037"
assert clean_text(text) == ""
text = "150集链接：http://study.163.com/course/courseMain.htm?courseId=1459037 全套150集,分初中级:循序渐进(完结)学习目的：为了让自己更有料些,不求于他人,就是这么简单.学习内容：根据实际案例进行解说,直击效果,要的也是这么简单.学习对象：1.从零开始学习透视表。2.需要交互式的汇报人员。---------------------------------------------数据汇总再也不求人了,全网易最全的EXCEL透视表学习对象!简单的,基础的,中级的,深入的基本上都涉及到.基础:从插入到设计选项卡,从排序到组合,从样式到打印,从动态到计算项,全部呈现出来.使用实操深入解析各项功能性的使用.中级:从动态源头切片器到图表,从导入外部数据到PowerPivot及view,从SQL到MicrosoftQuery,展示了它的更高级功能.产出:以案例最终呈现出大家经常看到的一页纸Dashboard的报告呈现,这应该都是学友需要看到的成品效果.------------------(其它章节连接地址:http://study.163.com/u/eeshang)"
text_a = "150集链接：全套150集,分初中级:循序渐进(完结 )学习目的：为了让自己更有料些,不求于他人,就是这么简单 .学习内容：根据实际案例进行解说,直击效果,要的也是这么 简单.学习对象：1.从零开始学习透视表。2.需要交互式的 汇报人员。----------------------- ----------------------数据汇总再也 不求人了,全网易最全的EXCEL透视表学习对象!简单的, 基础的,中级的,深入的基本上都涉及到.基础:从插入到设计 选项卡,从排序到组合,从样式到打印,从动态到计算项,全部 呈现出来.使用实操深入解析各项功能性的使用.中级:从动态 源头切片器到图表,从导入外部数据到PowerPivot及 view ,从SQL到Micro soft Query,展示了它的 更高级功能.产出:以案例最终呈现出大家经常看到的一页纸D ash board 的报告呈现,这应该都是学友需要看到的成品效果.- -----------------(其它章节连接地址:)"
assert clean_text(text) == text_a
print("All Pass [CLEAN]")
