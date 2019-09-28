using System;

public class Quine { public static void Main() {
    string t = "    ";
    string s = "string ";
    char n = '\n';
    char q = '"';
    char b = '\\';
    string l1 = "using System;";
    string l2 = "public class Quine { public static void Main() {";
    string l3 = "} }";
    string l4 = "Console.WriteLine(l5, t, s, n, q, l1, l2, l3, l4, l5, b);";
    string l5 = "{4}{2}{2}{5}{2}{0}{1}t = {3}{0}{3};{2}{0}{1}s = {3}{1}{3};{2}{0}char n = '{9}n';{2}{0}char q = '{3}';{2}{0}char b = '{9}{9}';{2}{0}{1}l1 = {3}{4}{3};{2}{0}{1}l2 = {3}{5}{3};{2}{0}{1}l3 = {3}{6}{3};{2}{0}{1}l4 = {3}{7}{3};{2}{0}{1}l5 = {3}{8}{3};{2}{2}{0}{7}{2}{6}";

    Console.WriteLine(l5, t, s, n, q, l1, l2, l3, l4, l5, b);
} }