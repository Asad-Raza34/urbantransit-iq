import java.lang.reflect.Method;

public class JniBindCheck {
    public static void main(String[] args) throws Exception {
        System.setProperty("hadoop.home.dir", "C:/Users/HP/Desktop/UrbanTransit IQ/hadoop");
        String dll = "C:/Users/HP/Desktop/UrbanTransit IQ/hadoop/bin/hadoop.dll";
        try {
            System.load(dll);
            System.out.println("EXPLICIT_SYSTEM_LOAD_OK");
        } catch (Throwable t) {
            System.out.println("EXPLICIT_SYSTEM_LOAD_FAILED -> " + t);
        }
        Class<?> c = Class.forName("org.apache.hadoop.io.nativeio.NativeIO$Windows");
        Method m = c.getDeclaredMethod("access0", String.class, int.class);
        m.setAccessible(true);
        try {
            Object out = m.invoke(null, "C:/Users/HP/Desktop/UrbanTransit IQ/hadoop/bin/winutils.exe", 0);
            System.out.println("ACCESS0_BOUND_AND_RAN -> " + out);
        } catch (java.lang.reflect.InvocationTargetException e) {
            System.out.println("ACCESS0_BOUND_BUT_FAILED -> " + e.getCause());
        } catch (java.lang.UnsatisfiedLinkError e) {
            System.out.println("BIND_FAILED -> " + e.getMessage());
        }
    }
}
