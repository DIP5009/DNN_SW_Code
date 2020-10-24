#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/io.h>
#include <linux/interrupt.h>

#include <linux/cdev.h>
//#include <linux/fs.h>
#include <linux/workqueue.h>

#include <linux/dma-mapping.h>

#include <linux/of_address.h>
#include <linux/of_device.h>
#include <linux/of_platform.h>

/* Standard module information, edit as appropriate */
MODULE_LICENSE("GPL");

#define DRIVER_NAME "cnn"

#define START 0
#define LAYER_INFO_0 1
#define LAYER_INFO_1 3
#define LAYER_INFO_2 4
#define LAYER_INFO_3 5
#define LAYER_INFO_4 6
#define R_ERROR_ADDR 7
#define W_ERROR_ADDR 8
#define R_ERROR_TYPE 9
#define W_ERROR_TYPE 10
#define MEM_ALLOC    11
#define MEM_FREE     12

struct cnn_local {
	unsigned long mem_start;
	unsigned long mem_end;
	void __iomem *base_addr;

	struct device *plat_dev_p; //Pointer of Platform Device

	//Char Device
	struct device *char_dev_p; //Pointer of Char Device
	dev_t dev_node;
	struct cdev cdev;
	struct class *class_p;

	//Interrupt
	int irq;
	struct completion cmp;

    //DRAM Pointer
    dma_addr_t   phys_addr;
    void         *virt_addr;
};

static long ioctl(struct file *file_p, unsigned int cmd, unsigned long arg)
{
    struct cnn_local *lp = (struct cnn_local *)file_p->private_data;
    unsigned long timeout;
    volatile unsigned int *tmp = (unsigned int *)lp->base_addr;

    switch(cmd){
    case START:
        timeout = msecs_to_jiffies(15000); //Max wait : 10 second
        init_completion(&lp->cmp);
        tmp[0] = 0;
        tmp[0] = 1;
        timeout = wait_for_completion_interruptible_timeout(&lp->cmp, timeout);
        if(timeout == 0)
            printk("Timeout\n");
        if(tmp[8] != 0 || tmp[9] != 0){
            return -1;
        }
        tmp[0] = 0;
        break;
    case LAYER_INFO_0:
        tmp[1] = arg;
        break;
    case LAYER_INFO_1:
        tmp[2] = arg;
        break;
    case LAYER_INFO_2:
        tmp[3] = arg;
        break;
    case LAYER_INFO_3:
        tmp[4] = arg;
        break;
    case LAYER_INFO_4:
        tmp[5] = arg;
        break;
    case R_ERROR_ADDR:
        return tmp[6];
    case W_ERROR_ADDR:
        return tmp[7];
    case R_ERROR_TYPE:
        return tmp[8];
    case W_ERROR_TYPE:
        return tmp[9];
    case MEM_ALLOC:
        lp->virt_addr = dma_alloc_coherent(lp->plat_dev_p, arg , &lp->phys_addr, GFP_KERNEL);
        printk("Allocate uncached memory at physical address = %lx, virtual address = %lx\n", lp->phys_addr, lp->virt_addr);
        if (!lp->virt_addr) {
            dev_err(lp->plat_dev_p, "Allocating uncached memory failed.\n");
            return -1;
        }
        return lp->phys_addr;
    case MEM_FREE:
        if (!lp->virt_addr) {
            dev_err(lp->plat_dev_p, "Releasing uncached memory failed.\n");
            return -1;
        }
        dma_free_coherent(lp->plat_dev_p, arg, lp->virt_addr, lp->phys_addr);
        break;
    default:
        printk("  [error] Sobel Driver: No Such Command\n");
        return -2;
	}
	return 0;
}

static int open(struct inode *ino, struct file *file_p)
{
	file_p->private_data = container_of(ino->i_cdev, struct cnn_local, cdev);
	return 0;
}

static int mmap(struct file *file_p, struct vm_area_struct *vma)
{
	struct cnn_local *lp = (struct cnn_local *)file_p->private_data;
	return dma_mmap_coherent(lp->plat_dev_p, vma,
		lp->virt_addr, lp->phys_addr, vma->vm_end - vma->vm_start);
}

static struct file_operations dm_fops = {
	.owner    = THIS_MODULE,
	.open     = open,
	.mmap	  = mmap,
	.unlocked_ioctl = ioctl,
};

static irqreturn_t cnn_irq(int irq, void *lp)
{
	struct cnn_local *lpp = (struct cnn_local *)lp;
	complete(&lpp->cmp);
	return IRQ_HANDLED;
}

static int cdevice_init(struct cnn_local *lp)
{
	int rc;
	static struct class *local_class_p = NULL;
	/* Allocate a character device from the kernel for this driver.
	 */
	rc = alloc_chrdev_region(&lp->dev_node, 0, 1, DRIVER_NAME);
	if (rc) {
		dev_err(lp->plat_dev_p, "unable to get a char device number\n");
		return rc;
	}
	/* Initialize the device data structure before registering the character 
	 * device with the kernel.
	 */
	cdev_init(&lp->cdev, &dm_fops);
	lp->cdev.owner = THIS_MODULE;

	rc = cdev_add(&lp->cdev, lp->dev_node, 1);
	if (rc) {
		dev_err(lp->plat_dev_p, "unable to add char device\n");
		goto init_error1;
	}
	/* Only one class in sysfs is to be created for multiple channels,
	 * create the device in sysfs which will allow the device node
	 * in /dev to be created
	 */
	if (!local_class_p) {
		local_class_p = class_create(THIS_MODULE, DRIVER_NAME);
		if (IS_ERR(lp->plat_dev_p->class)) {
			dev_err(lp->plat_dev_p, "unable to create class\n");
			rc = -1;
			goto init_error2;
		}
	}
	lp->class_p = local_class_p;
	/* Create the device node in /dev so the device is accessible
	 * as a character device
	 */
	lp->char_dev_p = device_create(lp->class_p, NULL,
					  	 lp->dev_node, NULL, DRIVER_NAME);
	if (IS_ERR(lp->plat_dev_p)) {
		dev_err(lp->plat_dev_p, "unable to create the char device\n");
		goto init_error3;
	}
	return 0;

init_error3:
	class_destroy(lp->class_p);

init_error2:
	cdev_del(&lp->cdev);

init_error1:
	unregister_chrdev_region(lp->dev_node, 1);
	return rc;
}

static void cdevice_exit(struct cnn_local *lp)
{
	if (lp->char_dev_p) {
		device_destroy(lp->class_p, lp->dev_node);
		class_destroy(lp->class_p);
		cdev_del(&lp->cdev);
		unregister_chrdev_region(lp->dev_node, 1);
	}
}

static int cnn_probe(struct platform_device *pdev)
{
	struct resource *r_irq; /* Interrupt resources */
	struct resource *r_mem; /* IO mem resources */
	struct device *dev = &pdev->dev;
	struct cnn_local *lp = NULL;

	int rc = 0;
	dev_info(dev, "Device Tree Probing\n");
	/* Get iospace for the device */
	r_mem = platform_get_resource(pdev, IORESOURCE_MEM, 0);
	if (!r_mem) {
		dev_err(dev, "invalid address\n");
		return -ENODEV;
	}
	lp = (struct cnn_local *) kmalloc(sizeof(struct cnn_local), GFP_KERNEL);
	if (!lp) {
		dev_err(dev, "Cound not allocate cnn device\n");
		return -ENOMEM;
	}
	dev_set_drvdata(dev, lp);
	lp->mem_start = r_mem->start;
	lp->mem_end = r_mem->end;

	if (!request_mem_region(lp->mem_start,
				lp->mem_end - lp->mem_start + 1,
				DRIVER_NAME)) {
		dev_err(dev, "Couldn't lock memory region at %p\n",
			(void *)lp->mem_start);
		rc = -EBUSY;
		goto error1;
	}

	lp->base_addr = ioremap(lp->mem_start, lp->mem_end - lp->mem_start + 1);
	if (!lp->base_addr) {
		dev_err(dev, "cnn: Could not allocate iomem\n");
		rc = -EIO;
		goto error2;
	}

	/* Get IRQ for the device */
	r_irq = platform_get_resource(pdev, IORESOURCE_IRQ, 0);
	if (!r_irq) {
		dev_info(dev, "no IRQ found\n");
		dev_info(dev, "cnn at 0x%08x mapped to 0x%08x\n",
			(unsigned int __force)lp->mem_start,
			(unsigned int __force)lp->base_addr);
		return 0;
	}
	lp->irq = r_irq->start;
	rc = request_irq(lp->irq, &cnn_irq, IRQF_SHARED | IRQF_TRIGGER_RISING, DRIVER_NAME, lp);
	if (rc) {
		dev_err(dev, "testmodule: Could not allocate interrupt %d.\n",
			lp->irq);
		goto error3;
	}

	dev_info(dev,"cnn at 0x%08x mapped to 0x%08x, irq=%d\n",
		(unsigned int __force)lp->mem_start,
		(unsigned int __force)lp->base_addr,
		lp->irq);
    rc = cdevice_init(lp);
    if(rc) {
	    dev_err(dev, "Char device initial failed!\n");
	    goto error3;
	}

	return 0;
error3:
	free_irq(lp->irq, lp);
error2:
	release_mem_region(lp->mem_start, lp->mem_end - lp->mem_start + 1);
error1:
	kfree(lp);
	dev_set_drvdata(dev, NULL);
	return rc;
}

static int cnn_remove(struct platform_device *pdev)
{
	struct device *dev = &pdev->dev;
	struct cnn_local *lp = dev_get_drvdata(dev);
    cdevice_exit(lp);
	free_irq(lp->irq, lp);
	iounmap(lp->base_addr);
	release_mem_region(lp->mem_start, lp->mem_end - lp->mem_start + 1);
	kfree(lp);
	dev_set_drvdata(dev, NULL);
	return 0;
}

#ifdef CONFIG_OF
static struct of_device_id cnn_of_match[] = {
	{ .compatible = "xlnx,CNN-Yolo-1.0", },
	{ /* end of list */ },
};
MODULE_DEVICE_TABLE(of, cnn_of_match);
#else
# define cnn_of_match
#endif


static struct platform_driver cnn_driver = {
	.driver = {
		.name = DRIVER_NAME,
		.owner = THIS_MODULE,
		.of_match_table	= cnn_of_match,
	},
	.probe		= cnn_probe,
	.remove		= cnn_remove,
};

static int __init cnn_init(void)
{
    printk("<1>Hello module world.\n");
	return platform_driver_register(&cnn_driver);
}


static void __exit cnn_exit(void)
{
	platform_driver_unregister(&cnn_driver);
}

module_init(cnn_init);
module_exit(cnn_exit);
