import { PageIntro, NextPage } from "@/components/site/PageIntro";
import { Channels } from "@/components/site/ChannelsSection";
import { pageMetadata } from "@/components/site/pageMeta";

export const metadata = pageMetadata({
  title: "Channels",
  description:
    "One BonClay agent on the phone, in a website widget, in meetings and on WhatsApp, with an honest note on how far each channel has been tested.",
  path: "/channels",
});

export default function ChannelsPage() {
  return (
    <>
      <PageIntro title="Where people can reach your agent.">
        Set an agent up once, then reach it by phone, in meetings, on your website and on WhatsApp. Each channel says how far it has been tested.
      </PageIntro>
      <Channels />
      <NextPage from="/channels" />
    </>
  );
}
